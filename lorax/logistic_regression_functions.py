"""Functions for Logistic Regression individual feature importance."""


def get_contrib_list_LR(lr, sample, column_names):
    """
    Get list of feature contribution for Logistic Regression.

    In:
        - lr: sklearn LogisticRegression instance
        - sample: list of feature values
        - column_names: list of feature names
    Out:
        - contrib_list:
    """
    contrib_list = []

    for i in range(len(lr.coef_[0])):
        product = lr.coef_[0][i] * sample[i]
        contrib_list.append((column_names[i], product))

    contrib_list.append(("Intercept", lr.intercept_[0]))

    return contrib_list
