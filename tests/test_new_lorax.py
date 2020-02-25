import os
import sys
project_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(project_path)

import pandas as pd
# from pandas.testing import assert_frame_equal
import numpy as np
import random
from datetime import datetime
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from lorax.lorax import TheLorax
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

features = [x for x in columns if x not in ['entity_id', 'as_of_date', 'outcome']]

data = pd.DataFrame(data, columns=columns)

# Testing the independence from id_col, date_col, outcome
# data = data.drop(['entity_id', 'as_of_date', 'outcome'], axis=1)


n_estimators = 2
max_depth = 2
global_clf = RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    random_state=42).fit(X, y)

class TestLorax(unittest.TestCase):
    """Tests cases for Lorax."""

    def test_feature_importances(self):
        """Test calculated feature importances."""
        # Setting up lorax
        lrx = TheLorax(
            clf=global_clf, 
            column_names=features,
            test_mat=data,
            id_col='entity_id',
            date_col='as_of_date', 
            outcome_col='outcome')

        # without id_col (zero indexed)
        # lrx_out = lrx.explain_example_new(test_mat=data, idx=0, pred_class=1, graph=False)

        sample = data.loc[0, features].values

        pred_class = 0 # The label w.r.t the explanations are generated
        lrx_out = lrx.explain_example_new(
            sample=sample, 
            test_mat=None, 
            descriptive=True,
            idx=None, 
            pred_class=pred_class,
            num_features=10, 
            graph=False
        )

        feature1_contrib = lrx_out.contribution.loc['feature1']
        feature5_contrib = lrx_out.contribution.loc['feature5']

        print('Asserting feature importance scores...')
        # Test cases for correct feature importances
        if pred_class == 1:
            self.assertEqual(feature1_contrib, 0.04889021376498209)
            self.assertEqual(feature5_contrib, -0.31556073962118303)
        else:
            self.assertEqual(feature1_contrib, -0.04889021376498209)
            self.assertEqual(feature5_contrib, 0.31556073962118303)

        self.assertFalse('feature3' in lrx_out.contribution)

    def test_feature_stats(self):
        """Testing the data loader"""
        
        lrx = TheLorax(
            clf=global_clf, 
            column_names=features,
            test_mat=data,
            id_col='entity_id',
            date_col='as_of_date', 
            outcome_col='outcome'
        )

        st1 = lrx.populate_feature_stats(data[features])

        pd.testing.assert_frame_equal(st1, lrx.feature_stats)

    def test_descriptive_explanation_cases(self):
        """ 
            There are different methods to get a descriptive explanation
            This test asserts all those methods yield the same answer        
        """
        pass

    def test_old_vs_new_lorax(self):
        """
            Verifying that the new explain method is 
            generating the same explanations as before

            Note: This test was deprecated after verufying that the new explain instance 
            returned the same results as the old one. 
            The old method was emoved from the class
        """
        pass
        # lrx = TheLorax(
        #     clf=global_clf, 
        #     column_names=features,
        #     test_mat=data,
        #     id_col='entity_id',
        #     date_col='as_of_date', 
        #     outcome_col='outcome'
        # )

        # pred_class = 0 # The label w.r.t the explanations are generated
        # idx = 2
        # lrx_out_new = lrx.explain_example(
        #     sample=None, 
        #     test_mat=None, 
        #     descriptive=True,
        #     idx=idx, 
        #     pred_class=pred_class,
        #     num_features=10, 
        #     graph=False
        # )

        # lrx_out_old = lrx.explain_example(
        #     idx=idx,
        #     pred_class=pred_class,
        #     num_features=10,
        #     graph=False,
        #     how='features'
        # )

        # pd.testing.assert_frame_equal(lrx_out_new, lrx_out_old)


    def test_explanation_patterns(self):
        """
            Testing whether the explanations interms of 
            feature patterns are generated correctly
        """
        pass

if __name__ == '__main__':
    print(data.columns.values)
    unittest.main()
    
