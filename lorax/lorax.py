"""Class for Lorax."""
import re
import logging
import pandas as pd
from math import sqrt
from scipy import stats

from lorax.utils import *
from lorax.random_forest_functions import get_contrib_list_RF
from lorax.logistic_regression_functions import get_contrib_list_LR

from IPython.core.display import HTML, display

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


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
        clf (sklearn classifier) The classifier to be explained, e.g.,
            sklearn.ensemble.RandomForestClassifier
        test_mat (pandas.DataFrame) The test matrix containing all examples to be
            explained. If `id_col=None` (the default), the id for referencing entities
            must be set as this dataframe's index.
        id_col (str) The column name for the entity id in the test matrix. If `None`
            (the default), the test matrix must be indexed by the entity id.
        date_col (str) The date column in the matrix (default: `as_of_date`)
        outcome_col (str) The outcome column in the matrix (default: `outcome`). To
            indicate that the test matrix has no labels, set `outcome_col=None`.
        name_patterns (list) An optional list of regex patterns or compiled regex
            objects to group together features for reporting contributions. If using,
            each feature name in the test matrix must match one and only one pattern.
    """

    def __init__(self, clf, test_mat, id_col=None,
                 date_col='as_of_date', outcome_col='outcome',
                 name_patterns=None):
        """
        Initialize Lorax.

        In:
            - ...
        Out:
            - ...
        """
        self.clf = clf
        self.combined_index = False

        df = test_mat.copy()

        if id_col is not None:
            # if ID isn't already the index
            df.set_index(id_col, inplace=True)
            if type(id_col) in [list, tuple]:
                self.combined_index = True

        # exclude non-feature columns (date, outcome if present)
        if date_col in id_col:
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
            {'pred': [p[1] for p in self.clf.predict_proba(self.X_test.values)]},
            index=self.X_test.index
            )

        # For classifiers with intercepts, we add the intercept as a "feature"
        if hasattr(self.clf, 'intercept_'):
            self.X_test["Intercept"] = [self.clf.intercept_[0] for i in range(len(self.X_test))]

        # pre-calcuate feature distribution statistics for each feature
        self._populate_feature_stats()

    def _populate_feature_stats(self):
        """Setter function for feature distribution statistics.

        Pre-calculates the feature distribution information from the test matrix, including
        type (continuous or binary), mean, median, 5th & 95th percentiles, standard deviation.
        """
        fstats = pd.DataFrame(columns=['feature', 'type', 'mean', 'stdev', 'median', 'p5', 'p95'])
        dtypes = self.X_test.dtypes
        for col in self.column_names:
            feat = self.X_test[col]
            d = {'feature': col,
                 'mean': feat.mean(),
                 'median': feat.median(),
                 'p5': feat.quantile(0.05),
                 'p95': feat.quantile(0.95),
                 'mean_pctl': stats.percentileofscore(feat, feat.mean())/100.0}
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

    def set_name_patterns(self, name_patterns):
        """Map regex patterns to column names.

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

    def _plot_graph(self, idx, pred_class, score,
                    num_features, contrib_df, how):
        """
        Plot feature importances.

        In:
            - idx: index for example
            - pred_class: predicted class
            - score: score for predicted class
            - num_features: number of top features to show
            - contrib_df: pandas DF with feature contributions
            - how: (str) Whether to calculate feature contributions at
                            the level of individual features
        Out:
            - plot
        """
        h3_str = '<h3>Explanations for example {} with predicted class={} ' +\
                 '(score for {}: {:.4f})</h3>'
        display(HTML(h3_str.format(idx, pred_class, pred_class, score)))
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

    def _build_contrib_df(self, mean_by_trees_list, idx, how):
        """
        Build contribution dataframe.

        In:
            - mean_by_trees_list (list):
            - idx: index for example
            - how: Whether to calculate feature contributions at
                    the level of individual features
        Out:
            - contrib_df (pandas DF)
        """
        contrib_df = pd.DataFrame(mean_by_trees_list,
                                  columns=['feature', 'contribution'])
        contrib_df.set_index('feature', inplace=True)

        # if we're using name patterns, aggregate columns to pattern level,
        # otherwise, join on column-level statistics (not available for pattern-level)
        if how == 'patterns':
            contrib_df = contrib_df.join(self.column_patterns, how='inner')
            contrib_df = contrib_df.groupby(['name_pattern'])['contribution'].sum().to_frame()
        else:
            contrib_df = contrib_df.join(self.feature_stats, how='left')

            # lookup the specific example's values
            for col in contrib_df.index.values:

                if self.combined_index:
                    example_value = self.X_test.loc[idx, col].values[0]
                else:
                    example_value = self.X_test.loc[idx, col]

                contrib_df.loc[col, 'example_value'] = example_value
                vals, pct_sco = self.X_test[col], example_value
                contrib_df.loc[col, 'example_pctl'] = stats.percentileofscore(vals, pct_sco) / 100.0

            contrib_df['z_score'] = 1.0 * (contrib_df['example_value'] - contrib_df['mean'])
            contrib_df['z_score'] = contrib_df['z_score'] / contrib_df['stdev']

        # sort the resulting dataframe in descending order by contribution
        contrib_df.sort_values('contribution', ascending=False, inplace=True)

        return contrib_df

    def _mean_of_feature_by_class(self, sample_id, feature_name, vectors, targets):
        """
        Get sample's value for feature, and mean of feature for class 0 and 1.

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
        """Plot a horizontal bar plot of feature contributions.

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
            ylabels = ['{} = {:.3f}'.format(f, v) for f, v in df['example_value'].iteritems()]
            _ = ax.set_yticklabels(ylabels)
        ax.set_xticks([])

        # add data labels for contributions
        rects = ax.patches
        labels = ['{:.3f}'.format(x) for x in df['contribution']]

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width(), rect.get_y()+height/2,
                    label, ha='left', va='center',
                    color='black', fontsize=14)

    def _plot_dists(self, df, num_features, ax):
        """Plot feature distributions on percentile scale.

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
        ax.set_xlim([-0.05, 1.05])

        # add z-score labels to right of plot
        rects = ax.patches
        labels = ['{:.2f}'.format(x) for x in df['z_score']]

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(1.05, rect.get_y()+height/2, label, ha='left', va='center',
                    color='black', fontsize=14)

        ax.text(1.05, num_features-0.3, 'z-value', ha='left',
                va='center', color='black', fontsize=14)

        # add a legend
        gray_patch = mpatches.Patch(color='lightgray', label=r'Test Set 5%-95%')
        gray_point = mlines.Line2D([], [], color='gray', marker='o', ms=10,
                                   linewidth=0, label='Test Set Mean')
        orange_point = mlines.Line2D([], [], color='#E59141', marker='o',
                                     ms=10, linewidth=0, label='This Value')
        ax.legend(
            handles=[gray_patch, gray_point, orange_point],
            fontsize=14, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.0),
            facecolor='white', edgecolor='None',
            handletextpad=0.2, columnspacing=0.2
        )

        # plot points for
        y_vals = list(range(num_features))
        ax.scatter(list(df['mean_pctl']), y_vals, color='gray', s=100, zorder=4)  # test set mean
        # example value
        ax.scatter(list(df['example_pctl']), y_vals, color='#E59141', s=100, zorder=5)
        # line for median
        ax.plot([0.5, 0.5], [-1, num_features], color='darkgray', linewidth=1, zorder=3)

        # cleanup
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_facecolor('white')
        ax.set_title('Feature Distributions', fontsize=16)

    def explain_example(self, idx, pred_class=None, num_features=10, graph=True, how='features'):
        """Graph or return individual feature importances for an example.

        This method is the primary interface for TheLorax to calculate individual feature
        importances for a given example (identified by `idx`). It can be used to either
        return a pandas DataFrame with contributions and feature distributions (if
        `graph=False`) or a graphical representation of the top `num_features` contributions
        (if `graph=True`, the default) for use in a jupyter notebook.

        Feature contributions can be calucalted either for all features separately (`how='features',
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
                (`how='features'`, the default) or using regex patterns (`how='patterns'`).
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
            raise ValueError('Must specify name patterns to aggregate over.' +
                             'Use TheLorax.set_name_patterns() first.')
        elif how not in ['features', 'patterns']:
            raise ValueError('How must be one of features or patterns.')

        # If we have MultiIndex, we need to sort
        if self.combined_index:
            self.preds.sort_index(level=0, inplace=True)
            self.X_test.sort_index(level=0, inplace=True)

        # score for this example for the positive class
        # using threshold of 0.5 if pred_class is not given as an argument
        score = self.preds.loc[idx, 'pred']
        if pred_class is None:
            pred_class = int(score >= 0.5)

        # feature values for this example
        sample = self.X_test.loc[idx, ].values
        if self.combined_index:
            sample = sample[0]

        if isinstance(self.clf, RandomForestClassifier):
            # Getting values for Random Forest Classifier
            return_tuple = get_contrib_list_RF(self.clf, sample, self.column_names)

            self.num_trees = return_tuple[0]
            self.global_score_dict = return_tuple[1]
            self.feature_dict = return_tuple[2]
            self.aggregated_dict = return_tuple[3]
            contrib_list = return_tuple[4]

        elif isinstance(self.clf, LogisticRegression):
            # Getting values for Random Forest Classifier
            contrib_list = get_contrib_list_LR(self.clf, sample, self.column_names)

        # TODO: handle this more elegantly for multiclass problems
        # We need to flip the sign of the scores.
        if pred_class == 0:
            score = 1.0 - score
            contrib_list = [(feature, score * -1) for feature, score in contrib_list]

        logging.info('Used predicted class {} for example {}, score={}'.format(pred_class,
                                                                               idx,
                                                                               score))

        # sorting in descending order by contribution then by feature name in the case of ties
        contrib_list.sort(key=lambda x: (x[1] * -1, x[0]))

        # drop the results into a dataframe to append on other information
        contrib_df = self._build_contrib_df(contrib_list, idx, how)

        # adding overall feature importance from model level
        overall_importance = []
        for i in range(len(self.column_names)):
            if isinstance(self.clf, LogisticRegression):
                overall_importance.append((self.column_names[i], self.clf.coef_[0][i]))
            else:
                overall_importance.append((self.column_names[i], self.clf.feature_importances_[i]))

        updated_list = add_overall_feature_importance(contrib_list,
                                                      overall_importance)
        updated_columns = ['feature', 'sample_rank', 'overall_imp', 'overall_rank', 'rank_change']

        contrib_df = contrib_df.join(pd.DataFrame(data=updated_list,
                                                  columns=updated_columns).set_index('feature'))

        if graph:
            self._plot_graph(idx, pred_class, score,
                             num_features, contrib_df, how)
        else:
            return contrib_df

    def speak_for_the_trees(self, id, pred_class=None, num_features=20, graph=True, how='features'):
        """Explain an example's score.

        This method is just a synonym for `explain_example()` because TheLorax has to be able
        to speak for the trees.
        """
        return self.explain_example(id, pred_class, num_features, graph, how)
