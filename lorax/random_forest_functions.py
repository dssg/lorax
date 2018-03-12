"""Functions for Random Forest feature importance."""

import logging
import numpy as np


def get_score(scores, node_idx):
    """Get score at node."""
    x, y = scores[node_idx][0]
    return y / (x + y)


def get_descriptive_dict(input_array, num_trees, mean_diff=False):
    """
    Get dict with descriptive statistics.

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
                      # sum of diffs over total number of trees
                      "mean_over_n_trees": sum(input_array) / num_trees,
                      "std": std,
                      "max_value": max_value,
                      "min_value": min_value
                      }

    return diff_list_dict


def get_dict_of_score_differences(sample,
                                  score_diff_dict,
                                  previous_feature,
                                  previous_score,
                                  node_id,
                                  feature,
                                  threshold,
                                  children_left,
                                  children_right,
                                  scores,
                                  column_names):
        """
        Get dictionary of score differences. Used to build global dict for feature importances.

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
        current_score = get_score(scores, node_id)
        current_score_diff = current_score - previous_score

        if node_id != 0:  # we don't want to add 0 diff from first node.
            if previous_feature in score_diff_dict:
                score_diff_dict[previous_feature].append(current_score_diff)
            else:
                score_diff_dict[previous_feature] = [current_score_diff]
        else:
            logging.info("Node id is 0. We do NOT add score_diff.")

        # now we need to find out if we have to go deeper
        current_feature_idx = feature[node_id]
        if current_feature_idx < 0:
            # We arrived at a leaf.
            return None
        else:
            current_feature = column_names[current_feature_idx]
            current_feature_value = sample[current_feature_idx]

            left_children_index = children_left[node_id]
            right_children_index = children_right[node_id]

            if current_feature_value <= threshold[node_id]:
                next_node_id = left_children_index
            else:
                next_node_id = right_children_index

        return score_diff_dict, current_feature, current_score, next_node_id


def build_global_dict(rf, sample, column_names, num_trees):
    """
    Build global dict for feature importances.

    In:
        - rf:
        - sample: the sample for which we want to get feature importances.
    Out:
        - (dict) of form: {tree_index: {'feature_1':[0.1, 0.2], 'feature_2':[0.7]}}
    """
    global_score_dict = {}  # {tree_index: {'feature_1':[0.1, 0.2], 'feature_2':[0.7]}}

    # loop through all trees
    for idx, estimator in enumerate(rf.estimators_):
        if ((idx + 1) % 100) == 0:
            logging.info("Starting work on tree {}/{}".format(idx + 1, num_trees))

        # Getting tree's values
        features = estimator.tree_.feature
        threshold = estimator.tree_.threshold
        # If feature's value <= threshold, go here
        children_left = estimator.tree_.children_left
        # If feature's value > threshold, go here
        children_right = estimator.tree_.children_right
        # 2d array of scores at each node/leaf.
        scores = estimator.tree_.value

        # Initializing start values.
        next_node_id = 0
        current_feature = column_names[features[next_node_id]]
        current_score = get_score(scores, next_node_id)
        score_diff_dict = {}

        need_to_continue = True
        while need_to_continue:
            result_tuple = get_dict_of_score_differences(sample,
                                                         score_diff_dict,
                                                         current_feature,
                                                         current_score,
                                                         next_node_id,
                                                         features,
                                                         threshold,
                                                         children_left,
                                                         children_right,
                                                         scores,
                                                         column_names)
            if result_tuple:
                score_diff_dict, current_feature, current_score, next_node_id = result_tuple
            else:
                need_to_continue = False

        global_score_dict[idx] = score_diff_dict

    return global_score_dict


def aggregate_feature_scores_across_trees(global_score_dict):
    """
    Do the first aggregation across trees.

    You might want to consider following this function with aggregate_scores.
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
                # we need a new copy of diff_list so that we don't
                # manipulate the diff_list of the global dict.
                # we achieve this by calling list(diff_list)
                feature_dict[feature] = (list(diff_list), mean_diff_list)

    return feature_dict


def aggregate_scores(feature_dict, num_trees):
    """
    Aggregate scores for features.

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
        diff_list_dict = get_descriptive_dict(np.array(diff_list), num_trees)
        mean_diff_list_dict = get_descriptive_dict(np.array(mean_diff_list),
                                                   num_trees,
                                                   mean_diff=True)

        aggregated_dict[feature] = {"diff_list": diff_list_dict,
                                    "mean_diff_list": mean_diff_list_dict}
    return aggregated_dict


def get_contrib_list_RF(rf, sample, column_names):
    """
    Get list of feature contribution for Random Forests.

    In:
        - rf:
        - sample:
        - column_names:
    Out:
        - contrib_list:
    """
    num_trees = rf.n_estimators

    # calculate the individual feature contributions within each tree
    global_score_dict = build_global_dict(rf, sample, column_names, num_trees)

    # aggregate feature contributions across trees
    feature_dict = aggregate_feature_scores_across_trees(global_score_dict)

    # further aggregation of feature contributions (mean, std, etc.)
    aggregated_dict = aggregate_scores(feature_dict, num_trees)

    # we want features' mean_over_n_trees contributions
    contrib_list = []
    for feature, feature_dict in aggregated_dict.items():
        contrib_list.append((feature, feature_dict['diff_list']['mean_over_n_trees']))

    return num_trees, global_score_dict, feature_dict, aggregated_dict, contrib_list
