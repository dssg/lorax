"""Tests for Lorax."""

import random
import unittest
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from lorax import TheLorax
from lorax.utils import add_overall_feature_importance

random.seed(42)

# Preparing dataset for tests
X, y = datasets.make_classification(n_samples=10000, n_features=5,
                                    n_informative=3, n_redundant=2,
                                    random_state=42)

# Preparing test matrix
start, end = datetime(2017, 1, 1), datetime(2017, 12, 31)
as_of_dates = np.asarray([start + (end - start) * random.random() for i in range(X.shape[0])])
entity_ids = np.arange(1, X.shape[0] + 1)

data = np.append(X, y.reshape(y.shape[0], 1), axis=1)
data = np.append(as_of_dates.reshape(y.shape[0], 1), data, axis=1)
data = np.append(entity_ids.reshape(y.shape[0], 1), data, axis=1)

columns = ["entity_id", "as_of_date", "feature1", "feature2",
           "feature3", "feature4", "feature5", "outcome"]

data = pd.DataFrame(data=data, columns=columns)

n_estimators = 2
max_depth = 2

global_clf = RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    random_state=42).fit(X, y)


class TestLorax(unittest.TestCase):
    """Tests cases for Lorax."""

    def test_calculated_feature_importances(self):
        """Test calculated feature importances."""
        # Setting up lorax
        lrx = TheLorax(global_clf, data, id_col='entity_id')
        lrx_out = lrx.explain_example(idx=1, pred_class=1, graph=False)

        feature1_contrib = lrx_out.contribution.loc['feature1']
        feature5_contrib = lrx_out.contribution.loc['feature5']

        # Test cases for correct feature importances
        self.assertEqual(feature1_contrib, 0.04889021376498209)
        self.assertEqual(feature5_contrib, -0.31556073962118303)
        self.assertFalse('feature3' in lrx_out.contribution)

    def test_aggregated_dict(self):
        """Test aggregated_dict."""
        n_estimators = 5
        max_depth = 1

        # Setting up classifier
        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     random_state=42)
        clf = clf.fit(X, y)

        # Setting up lorax
        lrx = TheLorax(clf, data, id_col='entity_id')
        _ = lrx.explain_example(idx=1, pred_class=1, graph=False)

        # Max depth is 1. Number of split_occurences must be equal to
        # occurences_in_n_trees.
        for feature in lrx.aggregated_dict:
            split_occ = lrx.aggregated_dict[feature]['diff_list']['split_occurences']
            occ_trees = lrx.aggregated_dict[feature]['mean_diff_list']['occurences_in_n_trees']
            self.assertEqual(split_occ, occ_trees)

    def test_logistic_regression_importances(self):
        """Test feature contributions from logistic regression."""
        # Setting up classifier
        clf = LogisticRegression(C=1., solver='lbfgs')
        clf.fit(X, y)

        # Setting up lorax
        lrx = TheLorax(clf, data, id_col='entity_id')
        lrx_out = lrx.explain_example(idx=1, pred_class=1, graph=False)

        feature1_contrib = lrx_out.contribution.loc['feature1']
        feature5_contrib = lrx_out.contribution.loc['feature5']

        # Test cases for correct feature importances
        self.assertEqual(feature1_contrib, 2.186415806126551)
        self.assertEqual(feature5_contrib, -3.228614405467005)

        # Test case if we can recover lr prediction
        # Can't use all of sample because it now contains intercept as last element
        sample = lrx.X_test.loc[1, ].values[:-1]
        lr_pred = clf.predict_proba(sample.reshape(1, -1))[0][1]
        lrx_pred = 1 / (1 + np.exp(-lrx_out.contribution.sum()))

        self.assertEqual(lrx_pred, lr_pred)

    def test_size_global_dict(self):
        """Test the size of the global dict."""
        n_estimators = 3
        max_depth = 1

        # Setting up classifier
        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     random_state=42)
        clf = clf.fit(X, y)

        # Setting up lorax
        lrx = TheLorax(clf, data, id_col='entity_id')
        _ = lrx.explain_example(idx=1, pred_class=1, graph=False)

        # Checking if there as many entries, i.e., trees in
        # global_score_dict as number of estimators in forest
        self.assertEqual(len(lrx.global_score_dict), n_estimators)

        # Checking if every dict entry, i.e., tree has max_depth keys
        # Since max_depth=1, every tree dict should have only one entry
        for i in range(n_estimators):
            self.assertEqual(len(lrx.global_score_dict[i]), 1)

        # Checking if dicts for only feature in tree do not
        # have more than one entry
        for tree_idx, feat_dict in lrx.global_score_dict.items():
            self.assertEqual(len(feat_dict), 1)

    def test_add_overall_feature_importance(self):
        """Test function to add overall feature importance."""
        sample_importance = [('feature1', 0.2), ('feature2', 0.4)]
        overall_importance = [('feature1', 0.6), ('feature2', 0.1)]

        result = add_overall_feature_importance(sample_importance, overall_importance)
        true_result = [('feature2', 1, 0.1, 2, 1),
                       ('feature1', 2, 0.6, 1, -1)]

        for i in range(len(true_result)):
            self.assertTupleEqual(true_result[i], result[i])

        # Setting up lorax
        lrx = TheLorax(global_clf, data, id_col='entity_id')
        lrx_out = lrx.explain_example(idx=1, pred_class=1, graph=False)

        feature1_overall_imp = global_clf.feature_importances_[0]

        self.assertEqual(feature1_overall_imp, lrx_out.overall_imp.loc['feature1'])
        self.assertEqual(lrx_out.overall_rank.loc['feature2'], 3)
        self.assertEqual(lrx_out.rank_change.loc['feature5'], -2)

    def test_multiple_rows_per_entity_id(self):
        """Test support of multiple rows per entity_id."""
        # Setting up lorax
        # Getting output on test matrix with one row per entity_id
        lrx = TheLorax(global_clf, data, id_col='entity_id')
        lrx_out = lrx.explain_example(idx=1, pred_class=1, graph=False)

        # Changing test matrix so that the second row belongs
        # to entity_id 1 as well
        new_data = data.copy()
        new_data.entity_id[new_data.entity_id == 2] = 1

        # Checking that the output for original row of entity 1
        # remains the same when using combined index
        lrx = TheLorax(global_clf, new_data, id_col=['entity_id', 'as_of_date'])
        out_multi_rows = lrx.explain_example(idx=(1, '2017-08-21 18:01:57.040781'),
                                             pred_class=1,
                                             graph=False)

        self.assertTrue(lrx_out.equals(out_multi_rows))

        # Checking that the output for new row of entity 1
        # is not the same as the original one. It should not be
        # because the output is calculated on a different row.
        out_multi_rows_2nd_row = lrx.explain_example(idx=(1, '2017-01-10 02:29:38.247451'),
                                                     pred_class=1,
                                                     graph=False)

        self.assertFalse(lrx_out.equals(out_multi_rows_2nd_row))


if __name__ == "__main__":
    unittest.main(exit=False)
