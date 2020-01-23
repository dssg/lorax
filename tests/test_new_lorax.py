import os
import sys
project_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(project_path)

import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from lorax.the_lorax import TheLorax
from lorax.utils import add_overall_feature_importance


def test_lorax_breast_cancer():
    data_dict = load_breast_cancer()
    X = data_dict['data']
    y = data_dict['target']

    columns = data_dict['feature_names']

    data = pd.DataFrame(X, columns=columns)

    # model
    n_estimators = 2
    max_depth = 2

    global_clf = RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    random_state=42).fit(X, y)

    # Lorax
    lrx = TheLorax(
        clf=global_clf, 
        column_names=columns,
        id_col=None, date_col=None)

    sample = X[0, :]

    lrx.explain_example_new(sample=sample)

if __name__ == '__main__':
    test_lorax_breast_cancer()