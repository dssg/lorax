"""Tests for Lorax."""

import random
import unittest
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

from lorax import TheLorax

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


class TestLorax(unittest.TestCase):
    """Tests cases for Lorax."""

    def test_calculated_feature_importances(self):
        """Test calculated feature importances."""
        n_estimators = 2
        max_depth = 2

        # Setting up classifier
        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     random_state=42).fit(X, y)

        # Setting up lorax
        lrx = TheLorax(clf, data, id_col='entity_id')
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


if __name__ == "__main__":
    unittest.main(exit=False)
