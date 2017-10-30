
from math import sqrt
from scipy import stats
import pandas as pd
import numpy as np

import re
from itertools import product

import logging
from IPython.core.display import HTML, display

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

def patterns_from_config(feature_config, include_metrics=True):
    """
    Helper function to parse a triage feature config and return
    regex patterns that will match the features either to the
    level of column name or column name + aggregation function.
    """
    name_patterns = []
    for fg in feature_config:
        prefix = fg['prefix']
        groups = r'(%s)' % r'|'.join(fg['groups'])
        intervals = r'[^_]+'

        for cat in fg.get('categoricals', []):
            col = cat['column']
            pattern = r'^%s_%s_%s_%s_' % (prefix, groups, intervals, col)
            if include_metrics:
                for met in cat['metrics']:
                    agg_pattern = pattern + (r'(.*)_%s$' % met)
                    name_patterns.append(agg_pattern)
            else:
                name_patterns.append(pattern)

        for agg in fg.get('aggregates', []):
            quant = agg['quantity']
            if isinstance(quant, dict):
                # quantity was specified with a SQL query and name
                quant = list(quant.keys())[0]
            else:
                # quantity was specified as a column name
                quant = quant

            pattern = r'^%s_%s_%s_%s_' % (prefix, groups, intervals, quant)
            if include_metrics:
                for met in agg['metrics']:
                    # note there could be an imputation flag for aggregates
                    # (different than categoricals which use a NULL category)
                    agg_pattern = pattern + (r'%s(_imp)?$' % met)
                    name_patterns.append(agg_pattern)
            else:
                name_patterns.append(pattern)

    return name_patterns

def categorical_patterns_from_config(feature_config):
    """
    Helper function to parse a triage feature config and return
    regex patterns that will combine across categoricals as well
    as aggregates and their imputation flags.
    """

    # TODO: In theory this could be treated more like the general
    # "features" case where we also provide distribution statistics,
    # but doing so will require a little work to figure out how to
    # visualize the categoricals and imputated values sensibly...

    name_patterns = []
    for fg in feature_config:
        prefix = fg['prefix']
        groups = fg['groups']
        intervals = fg['intervals']

        for cat in fg.get('categoricals', []):
            col = cat['column']
            metrics = cat['metrics']

            for group, interval, metric in product(
                groups, intervals, metrics
                ):
                name_patterns.append(r'^%s_%s_%s_%s_(.*)_%s$' % (
                    prefix, group, interval, col, metric
                ))

        for agg in fg.get('aggregates', []):
            metrics = agg['metrics']
            quant = agg['quantity']
            if isinstance(quant, dict):
                # quantity was specified with a SQL query and name
                quant = list(quant.keys())[0]
            else:
                # quantity was specified as a column name
                quant = quant

            for group, interval, metric in product(
                groups, intervals, metrics
                ):
                name_patterns.append(r'^%s_%s_%s_%s_%s(_imp)?$' % (
                    prefix, group, interval, quant, metric
                )) 

    return name_patterns


class TheLorax(object):
    def __init__(self, rf, test_mat, id_col=None, 
                 date_col='as_of_date', outcome_col='outcome', 
                 name_patterns=None):
        self.rf = rf

        df = test_mat.copy()
        if id_col is not None:
            # if ID isn't already the index
            df.set_index(id_col, inplace=True)

        drop_cols = [date_col]
        if outcome_col is not None:
            drop_cols.append(outcome_col)
            self.y_test = df[outcome_col]
        else:
            self.y_test = None

        self.X_test = df.drop(drop_cols, axis=1)
        self.column_names = self.X_test.columns.values

        if name_patterns is not None:
            self.set_name_patterns(name_patterns)
        else:
            self.column_patterns = None

        self.preds = pd.DataFrame(
            {'pred': [p[1] for p in rf.predict_proba(self.X_test.values)]},
            index=self.X_test.index
            )

        self._populate_feature_stats()

    def _populate_feature_stats(self):
        fstats = pd.DataFrame(columns=['feature', 'type', 'mean', 'stdev', 'median', 'p5', 'p95'])
        dtypes = self.X_test.dtypes
        for col in self.column_names:
            feat = self.X_test[col]
            d = {
                'feature': col, 
                'mean': feat.mean(), 
                'median': feat.median(),
                'p5': feat.quantile(0.05),
                'p95': feat.quantile(0.95),
                'mean_pctl': stats.percentileofscore(feat, feat.mean())/100.0
                }
            if dtypes[col] == 'int' and feat.max() == 1:
                d.update({
                    'type': 'binary',
                    'stdev': sqrt(feat.mean() * (1 - feat.mean()))
                })
            else:
                d.update({
                    'type': 'continuous',
                    'stdev': feat.std()
                })
            fstats = fstats.append(d, ignore_index=True)

        fstats.set_index('feature', inplace=True)
        self.feature_stats = fstats

    def _get_score(self, scores, node_idx):
        x, y = scores[node_idx][0]
        return y / (x + y)

    def set_name_patterns(self, name_patterns):
        column_patterns = {}

        # compile the regex patterns if they aren't already
        if isinstance(name_patterns[0], str):
            name_patterns = [re.compile(pat) for pat in name_patterns]

        # iterate through columns then regex values to find matches
        for col in self.column_names:
            for regex in name_patterns:
                # if we find a match, make sure the column hasn't already matched a different
                # pattern and error out if so (why we don't break the inner loop upon matching)
                if regex.search(col) is not None:
                    if column_patterns.get(col) is not None:
                        raise ValueError('Two matching patterns for column {}: {} and {}'.format(
                            col, column_patterns[col], regex.pattern
                            ))
                    else:
                        column_patterns[col] = regex.pattern
            # after looping through all the regex patterns, if we haven't found a match, error out
            if column_patterns.get(col) is None:
                raise ValueError('No matching patterns for column {}:'.format(col))

        # convert to a dataframe to easily join onto the other results
        df = pd.DataFrame(list(column_patterns.items()), columns=['feature', 'name_pattern'])
        df.set_index('feature', inplace=True)

        self.column_patterns = df


    def _get_dict_of_score_differences(self, sample, score_diff_dict, previous_feature, previous_score,
                                      node_id, feature, threshold, children_left,
                                      children_right, scores):
        """
        Helper function to build global dict for feature importances.
        Called by _build_global_dict(...).

        In:
            - score_diff_dict:
            - scores:
            - previous_feature:
            - previous_score:
            - node_id:
            - feature: self.rf's array of features
            - children_left: self.rf's array of left children
            - children_right: self.rf's array of right children
            - threshold: self.rf's array of thresholds at splits
        Out:
            -
        """
        current_score = self._get_score(scores, node_id)
        current_score_diff = current_score - previous_score

        if node_id != 0: # we don't want to add 0 diff from first node.
            if previous_feature in score_diff_dict:
                score_diff_dict[previous_feature].append(current_score_diff)
            else:
                score_diff_dict[previous_feature] = [current_score_diff]

        # now we need to find out if we have to go deeper
        current_feature_idx = feature[node_id]
        if current_feature_idx < 0:
            # We arrived at a leaf.
            return None
        else:
            current_feature = self.column_names[current_feature_idx]
            current_feature_value = sample[current_feature_idx]

            left_children_index = children_left[node_id]
            right_children_index = children_right[node_id]

            if current_feature_value <= threshold[node_id]:
                next_node_id = left_children_index
            else:
                next_node_id = right_children_index

        return score_diff_dict, current_feature, current_score, next_node_id

    def _build_global_dict(self, sample):
        """
        Builds global dict for feature importances.
        In:
            - sample: the sample for which we want to get feature importances.
        Out:
            - (dict) of form: {tree_index: {'feature_1':[0.1, 0.2], 'feature_2':[0.7]}}
        """
        global_score_dict = {} #{tree_index: {'feature_1':[0.1, 0.2], 'feature_2':[0.7]}}

        n_trees = self.rf.n_estimators

        # loop through all trees
        for idx, estimator in enumerate(self.rf.estimators_):
            if ((idx + 1) % 100) == 0:
                logging.info("Starting work on tree {}/{}".format(idx + 1, n_trees))

            # Getting tree's values
            feature = estimator.tree_.feature
            threshold = estimator.tree_.threshold
            children_left = estimator.tree_.children_left # If feature's values less or equal than threshold, go here.
            children_right = estimator.tree_.children_right # If feature's values higher than threshold, go here.
            scores = estimator.tree_.value # 2d array of scores at each node/leaf.

            # Initializing start values.
            next_node_id = 0
            current_feature = self.column_names[feature[next_node_id]]
            current_score = self._get_score(scores, next_node_id)
            score_diff_dict = {}

            need_to_continue = True
            while need_to_continue:
                result_tuple = self._get_dict_of_score_differences(sample, score_diff_dict, current_feature,
                                                             current_score, next_node_id, feature,
                                                             threshold, children_left, children_right, scores)
                if result_tuple:
                    score_diff_dict, current_feature, current_score, next_node_id = result_tuple
                else:
                    need_to_continue = False

            global_score_dict[idx] = score_diff_dict
        return global_score_dict

    def _aggregate_feature_scores_across_trees(self, global_score_dict):
        """
        Does the first aggregation across trees. You might want to consider
        following this function with aggregate_scores.
        In:
            - global_score_dict: (dict) of form
                {tree_index: {'feature_1':[0.1, 0.2], 'feature_2':[0.7]}}
        Out:
            - feature_dict: (dict) of form
                {'feature_1': ([0.1, 0.2, 0.3], [0.1, 0.25])}
                where first list contains all score differences and
                second list contains mean score differences within a tree.
        """
        feature_dict = {}

        for tree_id, trees_dict in global_score_dict.items():
            for feature, diff_list in trees_dict.items():
                if feature in feature_dict:
                    total_diff_list, mean_diff_list = feature_dict[feature]
                    total_diff_list += diff_list
                    mean_diff_list.append(np.array(diff_list).mean())
                    feature_dict[feature] = (total_diff_list, mean_diff_list)
                else:
                    mean_diff_list = [np.array(diff_list).mean()]
                    feature_dict[feature] = (diff_list, mean_diff_list)

        return feature_dict

    def _get_descriptive_dict(self, input_array, num_trees, mean_diff=False):
        """
        In:
            - input_array: np array
            - num_trees: (int) of number of trees
            - mean_diff: (bool) whether or not input list is from mean_diff
        Out:
            - dict
        """
        mean = input_array.mean()
        std = input_array.std()
        max_value = input_array.max()
        min_value = input_array.min()

        if mean_diff:
            occurence_string = "occurences_in_n_trees"
        else:
            occurence_string = "split_occurences"

        diff_list_dict = {"input_list": input_array,
                          "mean": mean,
                          occurence_string: len(input_array),
                          "mean_over_n_trees": sum(input_array) / num_trees, # sum of diffs over total number of trees
                          "std": std,
                          "max_value": max_value,
                          "min_value": min_value
                         }

        return diff_list_dict


    def _aggregate_scores(self, feature_dict, num_trees):
        """
        Aggregates scores for features.
        In:
            - feature_dict: (dict) of form
            {'feature_1': ([0.1, 0.2, 0.3], [0.1, 0.25])}
            where first list contains all score differences and
            second list contains mean score differences within a tree.
        Out:
            - aggregated_dict: (dict) of form
            ...
        """
        aggregated_dict = {}

        for feature, diff_list_tuple in feature_dict.items():

            diff_list, mean_diff_list = diff_list_tuple
            diff_list_dict = self._get_descriptive_dict(np.array(diff_list), num_trees)
            mean_diff_list_dict = self._get_descriptive_dict(np.array(mean_diff_list), num_trees, mean_diff = True)

            aggregated_dict[feature] = {"diff_list": diff_list_dict,
                                        "mean_diff_list": mean_diff_list_dict}
        return aggregated_dict

    def _add_overall_feature_importance(self, sample_importance, overall_importance):
        """
        Builds new list showing feature and its importances for a sample and in overall
        model.

        In:
            - sample_importance: (list) of form
                [('feature1', 'importance'), ('feature2', 'importance'))]
            - overall_importance: (list) of form
                [('feature1', 'importance'), ('feature2', 'importance'))]
        Out:
            - [('feature1', 'sample_imp', 'sample_rank', 'overall_imp', 'overall_rank', 'rank_change')]
        """
        # TODO: add overall importances to output
        updated_list = []
        sorted_sample_importance = sorted(sample_importance, key=lambda x: x[1], reverse=True)
        sorted_overall_importance = sorted(overall_importance, key=lambda x: x[1], reverse=True)

        for sample_idx in range(len(sorted_sample_importance)):
            feature, importance = sorted_sample_importance[sample_idx]

            for overall_idx in range(len(sorted_overall_importance)):
                overall_feature, overall_importance = sorted_overall_importance[overall_idx]

                if feature == overall_feature:
                    updated_list.append((feature, importance, sample_idx + 1,
                                         overall_importance, overall_idx + 1,
                                         overall_idx - sample_idx))
                    break

        return updated_list

    def _mean_of_feature_by_class(self, sample_id, feature_name, vectors, targets):
        """
        Returns sample's value for feature, and mean of feature for class 0 and 1.
        In:
            - sample_id: (int) for row number of sample
            - feature_name: (str) name of feature that we want to get mean for
            - vectors: () input matrix for model containing training values
            - targets: (np array) of targets
            - self.column_names: (list) of feature names
        Out:
            - (sample_id_value, class_0_mean, class_1_mean)
        """
        # TODO: create a visualization to compare value with training set means by class
        feature_idx = self.column_names.index(feature_name)
        class_0_mean = vectors[targets == 0, feature_idx].mean()
        class_1_mean = vectors[targets == 1, feature_idx].mean()
        sample_id_value = vectors[sample_id, feature_idx]

        return sample_id_value, class_0_mean, class_1_mean

    def _plot_contribs(self, df, num_features, ax, include_values=True):
        df['contribution'].plot(kind='barh', color='#E59141', ax=ax, fontsize=14)
        ax.set_facecolor('white')
        ax.set_ylabel('')
        ax.set_title('Contribution to Predicted Class', fontsize=16)

        if include_values:
            # set y labels with feature name and value
            _ = ax.set_yticklabels(['{} = {:.3f}'.format(feat, val) for feat, val in df['example_value'].iteritems() ])
        ax.set_xticks([])

        # add data labels for contributions
        rects = ax.patches
        labels = ['{:.3f}'.format(x) for x in df['contribution']]

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width(), rect.get_y()+height/2, label, ha='left', va='center', 
                    color='black', fontsize=14)

    def _plot_dists(self, df, num_features, ax):

        # gray bars for 5% - 95% interval of the data
        pd.DataFrame(
                {'a': [0.9]*num_features}
            ).plot(kind='barh', left=[0.05]*num_features, color='lightgray', width=0.5, ax=ax)
        ax.set_xlim([-0.05,1.05])

        # add z-score labels to right of plot
        rects = ax.patches
        labels = ['{:.2f}'.format(x) for x in df['z_score']]

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(1.05, rect.get_y()+height/2, label, ha='left', va='center', 
                    color='black', fontsize=14)

        ax.text(1.05, num_features-0.3, 'z-value', ha='left', va='center', color='black', fontsize=14)

        # add a legend
        gray_patch = mpatches.Patch(color='lightgray', label=r'Test Set 5%-95%')
        gray_point = mlines.Line2D([], [], color='gray', marker='o', ms=10, linewidth=0, label='Test Set Mean')
        orange_point = mlines.Line2D([], [], color='#E59141', marker='o', ms=10, linewidth=0, label='This Value')
        ax.legend(
            handles=[gray_patch, gray_point, orange_point], 
            fontsize=14, ncol=3, loc='upper center', bbox_to_anchor=(0.5,0.0), 
            facecolor='white', edgecolor='None',
            handletextpad=0.2, columnspacing=0.2
        ) 

        # plot points for 
        y_vals = list(range(num_features))
        ax.scatter(list(df['mean_pctl']), y_vals, color='gray', s=100, zorder=4)  # test set mean
        ax.scatter(list(df['example_pctl']), y_vals, color='#E59141', s=100, zorder=5) # example value
        ax.plot([0.5,0.5], [-1, num_features], color='darkgray', linewidth=1, zorder=3) # line for median

        # cleanup
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_facecolor('white')
        ax.set_title('Feature Distributions', fontsize=16)


    def explain_example(self, idx, pred_class=None, num_features=10, graph=True, how='features'):
        # TODO: Categoricals can be handled using regex patterns, but this currently precludes
        # showing feature distribution information (since we don't know how to combine distributions
        # for arbitary feature groupings), but if just using patterns for categoricals/imputed flags
        # we should still be able to show relevant distribution info...

        if how == 'patterns' and self.column_patterns is None:
            raise ValueError('Must specify name patterns to aggregate over. Use TheLorax.set_name_patterns() first.')
        elif how not in ['features', 'patterns']:
            raise ValueError('how must be one of features or patterns.')

        score = self.preds.loc[idx, 'pred']
        if pred_class is None:
            pred_class = int(score >= 0.5)

        # use score for predicted class, so 1-score for class=0
        if pred_class == 0:
            score = 1.0 - score

        logging.info('using predicted class {} for example {}, score={}'.format(pred_class, idx, score))

        sample = self.X_test.loc[idx, ].values

        global_score_dict = self._build_global_dict(sample)

        num_trees = self.rf.n_estimators
        feature_dict = self._aggregate_feature_scores_across_trees(global_score_dict)
        aggregated_dict = self._aggregate_scores(feature_dict, num_trees)

        mean_by_trees_list = [] # sum of diffs over total number of trees

        for feat, dic in aggregated_dict.items():
            mean_by_trees_list.append((feat, dic['diff_list']['mean_over_n_trees']))
                
        if pred_class == 0:
            # We need to flip the sign of the scores.
            mean_by_trees_list = [(feature, score * -1) for feature, score in mean_by_trees_list]

        mean_by_trees_list.sort(key=lambda x: (x[1] * -1, x[0]))

        contrib_df = pd.DataFrame(mean_by_trees_list, columns=['feature', 'contribution'])
        contrib_df.set_index('feature', inplace=True)

        # if we're using name patterns, aggregate columns to pattern level,
        # otherwise, join on column-level statistics (not available for pattern-level)
        if how == 'patterns':
            contrib_df = contrib_df.join(self.column_patterns, how='inner')
            contrib_df = contrib_df.groupby(['name_pattern'])['contribution'].sum().to_frame()
        else:
            contrib_df = contrib_df.join(self.feature_stats, how='inner')
            # lookup the specific example's values
            for col in contrib_df.index.values:
                contrib_df.loc[col, 'example_value'] = self.X_test.loc[idx, col]
                contrib_df.loc[col, 'example_pctl'] = stats.percentileofscore(self.X_test[col], self.X_test.loc[idx, col])/100.0
            contrib_df['z_score'] = 1.0*(contrib_df['example_value'] - contrib_df['mean'])/contrib_df['stdev']

        # sort the resulting dataframe in descending order by contribution
        contrib_df.sort_values('contribution', ascending=False, inplace=True)

        if graph:
            display(HTML('<h3>Explanations for example {} with predicted class={} (score for {}: {:.4f})</h3>'.format(idx, pred_class, pred_class, score)))
            # subset to top features then sort ascending (since first will be at bottom of plots)
            df_subset = contrib_df.head(num_features).sort_values('contribution')
            if how == 'features':
                fig, ax = plt.subplots(1, 2, figsize=(14, 4.8*num_features/5))
                self._plot_contribs(df_subset, num_features, ax[0])
                self._plot_dists(df_subset, num_features, ax[1])
            elif how == 'patterns':
                fig, ax = plt.subplots(1, 1, figsize=(7, 4.8*num_features/5))
                self._plot_contribs(df_subset, num_features, ax, include_values=False)
            plt.show()

        else:
            return contrib_df

    def top_k_example_ids(self, k=10):
        # TODO: Handle ties better
        return list(self.preds.sort_values('pred', ascending=False).head(k).index)

    def bottom_k_example_ids(self, k=10):
        # TODO: Handle ties better
        return list(self.preds.sort_values('pred').head(k).index)

    def speak_for_the_trees(self, id, pred_class=None, num_features=20, graph=True, how='features'):
        return self.explain_example(id, pred_class, num_features, graph, how)
