"""Utils for Lorax."""

from itertools import product


def patterns_from_config(feature_config, include_metrics=True):
    """
    Parese a triage feature config and return
    regex patterns that will match the features either to the
    level of column name or column name + aggregation function.

    When using with `include_metrics=True` (the default), returned
    patterns will be of the form:
        categoricals: ^prefix_(group1|group2|...)_[^_]+_column_(.*)_metric$
        aggregates: ^prefix_(group1|group2|...)_[^_]+_quantity_metric(_imp)?$
    So, any feature with the same prefix, column/quantity, and aggregation
    metric will be treated as the same for aggregating contributions.

    When using with `include_metrics=False`, the returned patterns
    will be of the form:
        categoricals: ^prefix_(group1|group2|...)_[^_]+_column_
        aggregates: ^prefix_(group1|group2|...)_[^_]+_quantity_
    In this case, feature contribution will be aggregated at the level
    of prefix and column/quantity.

    Note that in either case, the returned patterns assume no column
    name truncation will take place between the config and the matrix.

    Arguments:
        feature_config (dict) A triage feature config for all of
            the features in the test matrix to be explained
        include_metrics (bool) When True, aggregation metrics
            (sum/avg/max/etc.) will be included in the pattern
            to match, making these distinct sets of features

    Returns:
        (list) A list of regular expression patterns to match the
        sets of features in the feature config

    """
    name_patterns = []
    for fg in feature_config:
        prefix = fg['prefix']
        groups = r'(%s)' % r'|'.join(fg['groups'])
        intervals = r'[^_]+'

        for cat in fg.get('categoricals', []):
            col = cat['column']
            pattern = r'^%s_%s_%s_%s_' % (prefix, groups, intervals, col)
            if include_metrics:
                for met in cat['metrics']:
                    agg_pattern = pattern + (r'(.*)_%s$' % met)
                    name_patterns.append(agg_pattern)
            else:
                name_patterns.append(pattern)

        for agg in fg.get('aggregates', []):
            quant = agg['quantity']
            if isinstance(quant, dict):
                # quantity was specified with a SQL query and name
                quant = list(quant.keys())[0]
            else:
                # quantity was specified as a column name
                quant = quant

            pattern = r'^%s_%s_%s_%s_' % (prefix, groups, intervals, quant)
            if include_metrics:
                for met in agg['metrics']:
                    # note there could be an imputation flag for aggregates
                    # (different than categoricals which use a NULL category)
                    agg_pattern = pattern + (r'%s(_imp)?$' % met)
                    name_patterns.append(agg_pattern)
            else:
                name_patterns.append(pattern)

    return name_patterns


def categorical_patterns_from_config(feature_config):
    """
    Helper function to parse a triage feature config and return
    regex patterns that will combine across categoricals as well
    as aggregates and their imputation flags, but otherwise treat
    all feature parameters as distinct.
    Here, returned patterns will be of the form:
        categoricals: ^prefix_group_interval_column_(.*)_metric$
        aggregates: ^prefix_group_interval_quantity_metric(_imp)?$
    Note that the returned patterns assume no column name truncation
    will take place between the config and the matrix.

    Arguments:
        feature_config (dict) A triage feature config for all of
            the features in the test matrix to be explained

    Returns:
        (list) A list of regular expression patterns to match the
        sets of features in the feature config

    """
    # TODO: In theory this could be treated more like the general
    # "features" case where we also provide distribution statistics,
    # but doing so will require a little work to figure out how to
    # visualize the categoricals and imputated values sensibly...

    name_patterns = []
    for fg in feature_config:
        prefix = fg['prefix']
        groups = fg['groups']
        intervals = fg['intervals']

        for cat in fg.get('categoricals', []):
            col = cat['column']
            metrics = cat['metrics']

            for group, interval, metric in product(groups,
                                                   intervals,
                                                   metrics):
                name_patterns.append(r'^%s_%s_%s_%s_(.*)_%s$' % (
                    prefix, group, interval, col, metric
                ))

        for agg in fg.get('aggregates', []):
            metrics = agg['metrics']
            quant = agg['quantity']
            if isinstance(quant, dict):
                # quantity was specified with a SQL query and name
                quant = list(quant.keys())[0]
            else:
                # quantity was specified as a column name
                quant = quant

            for group, interval, metric in product(groups,
                                                   intervals,
                                                   metrics):
                name_patterns.append(r'^%s_%s_%s_%s_%s(_imp)?$' % (
                    prefix, group, interval, quant, metric
                ))

    return name_patterns


def add_overall_feature_importance(sample_importance, overall_importance):
    """
    Build new list to compare feature importance for sample and in overall model.

    In:
        - sample_importance: (list) of form
            [('feature1', 'importance'), ('feature2', 'importance'))]
        - overall_importance: (list) of form
            [('feature1', 'importance'), ('feature2', 'importance'))]
    Out:
        - [('feature1', 'sample_rank',
            'overall_imp', 'overall_rank', 'rank_change')]
    """
    updated_list = []
    sorted_sample_importance = sorted(sample_importance, key=lambda x: x[1], reverse=True)
    sorted_overall_importance = sorted(overall_importance, key=lambda x: x[1], reverse=True)

    for sample_idx in range(len(sorted_sample_importance)):
        feature, importance = sorted_sample_importance[sample_idx]

        for overall_idx in range(len(sorted_overall_importance)):
            overall_feature, overall_importance = sorted_overall_importance[overall_idx]

            if feature == overall_feature:
                updated_list.append((feature, sample_idx + 1,
                                     overall_importance, overall_idx + 1,
                                     overall_idx - sample_idx))
                break

    return updated_list
