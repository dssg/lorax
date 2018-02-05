import re
import logging
import numpy as np
import pandas as pd

from lorax.utils import *

from math import sqrt
from scipy import stats
from itertools import product
from matplotlib import pyplot as plt

import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from IPython.core.display import HTML, display


class TheLorax(object):
    """TheLorax provides individual feature importances for random forest models.

    Given an sklearn random forest model object and test matrix (in the form of
    a pandas dataframe), `explain_example()` can be used to provide individual
    feature importances for a given entity in the test matrix. These feature
    contributions can be output in either a graphical format (for use in a
    jupyter notebook) or as an output dataframe.

    Currently, TheLorax has only been tested with random forests for binary
    classification problems, but future modifications could allow it to be used
    with other types of models and problems (regression/multinomial classification)

    Args:
        - rf (sklearn.ensemble.RandomForestClassifier): The classifier to be explained
        - test_mat (pandas.DataFrame): The test matrix containing all examples to be
            explained. If `id_col=None` (the default), the id for referencing entities
            must be set as this dataframe's index.
        - id_col (str): The column name for the entity id in the test matrix. If `None`
            (the default), the test matrix must be indexed by the entity id.
        - date_col (str): The date column in the matrix (default: `as_of_date`)
        - outcome_col (str): The outcome column in the matrix (default: `outcome`). To
            indicate that the test matrix has no labels, set `outcome_col=None`.
        - name_patterns (list): An optional list of regex patterns or compiled regex
            objects to group together features for reporting contributions. If using,
            each feature name in the test matrix must match one and only one pattern.
        - multiple_dates_per_id (bool): A bool to indicate whether or not test matrix contains
            multiple rows, i.e., dates, per entity_id. Default is False.

    """
    def __init__(self, rf, test_mat, id_col=None,
                 date_col='as_of_date', outcome_col='outcome',
                 name_patterns=None, multiple_dates_per_id=False):
        self.rf = rf

        df = test_mat.copy()

        index_columns=[]
        if id_col is not None:
            # if ID isn't already the index
            index_columns.append(id_col)

        # If we have multiple dates per entity, we need date_col
        # to be part of index as well.
        if multiple_dates_per_id:
            index_columns.append(date_col)

        if index_columns:
            df.set_index(index_columns, inplace=True)

        # exclude non-feature columns (date [depends on multiple_dates_per_id],
        # outcome if present)
        if multiple_dates_per_id:
            drop_cols = []
        else:
            drop_cols = [date_col]

        if outcome_col is not None:
            drop_cols.append(outcome_col)
            self.y_test = df[outcome_col]
        else:
            self.y_test = None

        self.X_test = df.drop(drop_cols, axis=1)
        self.column_names = self.X_test.columns.values

        # Register the regex patterns and associated columns if using
        if name_patterns is not None:
            self.set_name_patterns(name_patterns)
        else:
            self.column_patterns = None

        # predicted scores for the 1 class
        self.preds = pd.DataFrame(
            {'pred': [p[1] for p in rf.predict_proba(self.X_test.values)]},
            index=self.X_test.index
            )

        # pre-calcuate feature distribution statistics for the each feature
        self._populate_feature_stats()

    def _populate_feature_stats(self):
        """Setter function for feature distribution statistics

        Pre-calculates the feature distribution information from the test matrix, including
        type (continuous or binary), mean, median, 5th & 95th percentiles, standard deviation.
        """
        fstats = pd.DataFrame(columns=['feature', 'type', 'mean', 'stdev', 'median', 'p5', 'p95', 'mean_pctl'])
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
            # TODO: Currently detect binary columns as integers with max value of 1, but
            # might be better to just check cardinality directly and not rely on datatype
            if dtypes[col] == 'int' and feat.max() == 1:
                d.update({
                    'type': 'binary',
                    'stdev': sqrt(feat.mean() * (1 - feat.mean()))  # sqrt(p*(1-p)) for binary stdev
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
        """Map regex patterns to column names

        When using regular expressions to combine features into groups for aggregating
        contributions, this method will create a lookup dataframe from every feature
        in the test matrix to its matching pattern. Features that do not match any
        patterns (or match multiple patterns) will raise an exception.

        Note that this function will be called when the object is created if
        `name_patterns` was specified in the constructor or on an existing object by
        calling this method directly (this method may also be used to update the
        regex mapping associated with the object).

        Arguments:
            name_patters (list) A list of regex patterns or compiled regex objects to
            group together features for reporting contributions. If using, each feature
            name in the test matrix must match one and only one pattern.
        """
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
        """Plot a horizontal bar plot of feature contributions

        Arguments:
            df (pandas.DataFrame) The dataframe containing data to plot. Must be indexed
                on feature names and contain a "contribution" column. If `include_values`
                is True, must also include an "example_value" column.
            num_features (int) The number of top features being plotted
            ax (matplotlib.axes.Axes) The matplotlib axis object on which to plot the data
            include_values (bool) Whether to include information on the value of the
                feature for the current example in the axis ticks (default: True)
        """
        df['contribution'].plot(kind='barh', color='#E59141', ax=ax, fontsize=14)
        ax.set_facecolor('white')
        ax.set_ylabel('')
        ax.set_title('Contribution to Predicted Class', fontsize=16)

        # the option not to include values is provided for regex aggregations of contributions
        # where the entity doesn't have a single value for the set of features
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
        """Plot feature distributions on percentile scale

        This method will generate a plot to provide a quick look at the distribution of
        each of the top n features in the test set, including a few pieces of information:
            * 5th - 95th percentile range (gray bar)
            * mean (gray dot) relative to this range
            * entity value (orange dot) relative to this range
            * z-score for the entity value (data label)

        Arguments:
            df (pandas.DataFrame) The dataframe containing data to plot. Must be indexed
                on feature names and contain a "contribution" column. If `include_values`
                is True, must also include an "example_value" column.
            num_features (int) The number of top features being plotted
            ax (matplotlib.axes.Axes) The matplotlib axis object on which to plot the data
        """
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
        """Graph or return individual feature importances for an example

        This method is the primary interface for TheLorax to calculate individual feature
        importances for a given example (identified by `idx`). It can be used to either
        return a pandas DataFrame with contributions and feature distributions (if
        `graph=False`) or a graphical representation of the top `num_features` contributions
        (if `graph=True`, the default) for use in a jupyter notebook.

        Feature contributions can be calculated either for all features separately (`how='features',
        the default) or using regular expression patterns to group sets of features together
        (`how='patterns'`). When graphing contributions for all features, graphs will contain two
        components:
            1. A bar graph of the top num_features contributions to the example's score
            2. For each of these features, a graph showing the percentile for the feature's mean
               across the entire test set (gray dot), the percentile of the feature value for the
               example being explained (orange dot) and the z-score for that value
        When using regular expression patterns, the feature distribution information is omitted
        (from both graphical and dataframe outputs) as the contributions reflect aggregations over
        an arbitrary number and types of features.

        Arguments:
            idx (int) The entity id of the example we want to explain
            pred_class (int) The predicted class for the example (currently must be 1 or 0). The
                returned feature contributions will be taken relative to the score for this class.
                If None (the default), the predicted class will be assigned based on whether the
                example's score is above or below a threshold of 0.5.
            num_features (int) The number of features with the highest contributions to graph
                (ignored if `graph=False` in which case the entire set will be returned)
            graph (bool) Whether to graph the feature contributions or return a dataframe
                without graphing (default: True)
            how (str) Whether to calculate feature contributions at the level of individual features
                (`how='features'`, the default) or using regular expression patterns (`how='patterns'`).
                If using regex patterns, `name_patterns` must have been provided when the object
                was constructed or through calling `set_name_patterns()`.

        Returns:
            If `graph=False`, returns a pandas dataframe with individual feature contributions
            and (if using `how='features'`) feature distribution information
        """

        # TODO: Categoricals can be handled using regex patterns, but this currently precludes
        # showing feature distribution information (since we don't know how to combine distributions
        # for arbitary feature groupings), but if just using patterns for categoricals/imputed flags
        # we should still be able to show relevant distribution info...

        if how == 'patterns' and self.column_patterns is None:
            raise ValueError('Must specify name patterns to aggregate over. Use TheLorax.set_name_patterns() first.')
        elif how not in ['features', 'patterns']:
            raise ValueError("how must be one of 'features' or 'patterns'.")

        # score for this example for the positive class
        # and threshold and 0.5 if pred_class is not given as an argument
        score = self.preds.loc[idx, 'pred']
        if pred_class is None:
            pred_class = int(score >= 0.5)

        # use score for predicted class, so 1-score for class=0
        if pred_class == 0:
            score = 1.0 - score

        logging.info('using predicted class {} for example {}, score={}'.format(pred_class, idx, score))

        # feature values for this example
        sample = self.X_test.loc[idx, ].values

        # calculate the individual feature contributions
        global_score_dict = self._build_global_dict(sample)

        num_trees = self.rf.n_estimators
        feature_dict = self._aggregate_feature_scores_across_trees(global_score_dict)
        aggregated_dict = self._aggregate_scores(feature_dict, num_trees)

        mean_by_trees_list = [] # sum of diffs over total number of trees

        for feat, dic in aggregated_dict.items():
            mean_by_trees_list.append((feat, dic['diff_list']['mean_over_n_trees']))

        # TODO: handle this more elegantly for multiclass problems
        if pred_class == 0:
            # We need to flip the sign of the scores.
            mean_by_trees_list = [(feature, score * -1) for feature, score in mean_by_trees_list]

        # sorting in descending order by contribution then by feature name in the case of ties
        mean_by_trees_list.sort(key=lambda x: (x[1] * -1, x[0]))

        # drop the results into a dataframe to append on other information
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
        """Entities with the highest scores

        A quick helper function to get a list of the individuals with the highest scores

        Arguments:
            k (int) How many examples to return
        Returns:
            (list) Set of k entity ids with highest scores relative to the positive class
        """
        # TODO: Handle ties better
        return list(self.preds.sort_values('pred', ascending=False).head(k).index)

    def bottom_k_example_ids(self, k=10):
        """Entities with the lowest scores

        A quick helper function to get a list of the individuals with the lowest scores

        Arguments:
            k (int) How many examples to return
        Returns:
            (list) Set of k entity ids with lowest scores relative to the positive class
        """
        # TODO: Handle ties better
        return list(self.preds.sort_values('pred').head(k).index)

    def speak_for_the_trees(self, id, pred_class=None, num_features=20, graph=True, how='features'):
        """Explain an example's score

        This method is just a synonym for `explain_example()` because TheLorax has to be able
        to speak for the trees.
        """
        return self.explain_example(id, pred_class, num_features, graph, how)
