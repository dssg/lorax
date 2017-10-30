# The Lorax

Individual feature importances from random forests.

[![Build Status](https://travis-ci.org/dssg/lorax.svg?branch=master)](https://travis-ci.org/dssg/lorax)
[![codecov](https://codecov.io/gh/dssg/lorax/branch/master/graph/badge.svg)](https://codecov.io/gh/dssg/lorax)
[![codeclimate](https://codeclimate.com/github/dssg/lorax.png)](https://codeclimate.com/github/dssg/lorax)

Understanding the top features contributing to an individual example's model score can be important both for people building models to understand, debug, and improve their predictions and for people applying those models to trust and take appropriate actions based on their outputs. Overall, model-level feature importances can give a sense of what the modeling algorithm is surfacing as important in general but may have only mimimal relevance to what actually drives the score of any given individual. The Lorax helps to solve this problem by measuring feature contributions on the level of individual scores, currently developed and tested for the context of binary classification problems using Random Forest classifiers.

## Usage

### Basic Usage

`TheLorax` can be initialized with a trained random forest classifier (specifically, of class `sklearn.ensemble.RandomForestClassifier` and test set matrix (a pandas `DataFrame`), for instance:
```
lrx = TheLorax(rf, test_mat)
```

The resulting object can be used calculate feature contributions to an individual example's score using:
```
lrx.explain_example(example_id, pred_class=1, num_features=10, graph=True)
```

Using `graph=False` will simply return a dataframe with the feature contributions and distributions (for any feature with non-zero contribution to the score). For non-graphical output, `num_features` is ignored and the resulting dataframe contains all feature contributions.

When graphing, the output will contain two components:
1. A bar graph of the top `num_features` contributions to the example's score
1. For each of these features, a graph showing the percentile for the feature's mean across the entire test set (gray dot), the percentile of the feature value for the example being explained (orange dot) and the z-score for that value

### Regex Pattern Usage

You may want to look at the contributions of groups of features, which can be accomplished by providing regular expression patterns to combine feature names into sets:
```
lrx = TheLorax(rf, test_mat, name_patterns=list_of_regex)
```

Where `list_of_regex` could either by string regex patterns or compiled regex objects (each feature name must match one and only one pattern in the list). These patterns can also be called after creating a Lorax object with `lrx.set_name_patterns(list_of_regex)`.

Then the explanations can make use of these patterns by calling:
```
lrx.explain_example(example_id, pred_class=1, num_features=10, graph=True, how='patterns')
```

Explaining examples using name patterns will provide contributions grouped to the pattern level but note that feature distribution information isn't provided when aggregating to pattern levels.

## Feature Contribution Calculations

The basic method used here for calculating feature contributions is:

1. Traverse the trees with an individual example’s feature values (just as with making a prediction)
1. At every split, that example’s score changes based on the value it has for the feature split on
1. Keep track of these changes, associating them with the splitting feature
1. Sum the changes associated with each feature and divide by the number of trees, giving the contribution of that feature
1. The sum of these contributions will (roughly) be the difference between the example’s score and overall mean (roughly because of bootstrapping, assuming no balancing done in the bootstrap)
