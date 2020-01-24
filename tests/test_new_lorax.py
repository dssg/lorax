import os
import sys
project_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(project_path)

import pandas as pd
import numpy as np
import random
from datetime import datetime
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from lorax.the_lorax import TheLorax
from lorax.utils import add_overall_feature_importance

import unittest

# Data generation for classification
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

data = pd.DataFrame(data, columns=columns)

# Testing the independence from id_col, date_col, outcome
data = data.drop(['entity_id', 'as_of_date', 'outcome'], axis=1)


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
        lrx = TheLorax(
            global_clf, 
            column_names=columns,
            id_col=None,
            date_col=None, 
            outcome_col=None)

        # without id_col (zero indexed)
        # lrx_out = lrx.explain_example_new(test_mat=data, idx=0, pred_class=1, graph=False)

        sample = data.loc[0]
        lrx_out = lrx.explain_example_new(
            sample=sample, 
            test_mat=data, 
            idx=None, 
            pred_class=None, 
            graph=False)


        print(lrx_out)

        feature1_contrib = lrx_out.contribution.loc['feature1']
        feature5_contrib = lrx_out.contribution.loc['feature5']

        # Test cases for correct feature importances
        self.assertEqual(feature1_contrib, 0.04889021376498209)
        self.assertEqual(feature5_contrib, -0.31556073962118303)
        self.assertFalse('feature3' in lrx_out.contribution)


if __name__ == '__main__':
    # test_lorax_breast_cancer()
    unittest.main()