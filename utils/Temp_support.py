import numpy as np
import pandas as pd


from utils import Preprocess


def feature_padding(ds, features, preffix_patterns, encoding='utf-8', header=0, index_col=0):
    ## get suffix of features in given class
    classed_class_features = Preprocess.pattern_to_feature(ds, preffix_patterns, encoding=encoding)
    tmp = [list(map(lambda fc, pf=preffix: fc[len(pf) - 1:], feature_class)) for preffix, feature_class in zip(preffix_patterns, classed_class_features)]
    class_suffix = []
    for t in tmp:
        class_suffix.extend(t)
    class_suffix = list(set(class_suffix))
    # print('feature_padding: preffix_patterns = {}'.format(preffix_patterns))
    ## get features with mutually exclusive suffixs
    mut_exc_feature = []
    for suffix in class_suffix:
        for i, t in enumerate(tmp):
            if suffix in t:
                mut_exc_feature.append(preffix_patterns[i][1:] + suffix)
                break
        # if suffix in tmp[0]:
        #     mut_exc_feature.append(preffix_patterns[0][1:] + suffix)
        # elif suffix not in tmp[0]:
        #     mut_exc_feature.append(preffix_patterns[1][1:] + suffix if suffix in tmp[1] else preffix_patterns[2][1:] + suffix)
    return mut_exc_feature


def prefix_from_meta(flag_feature):
    ds = pd.read_csv('data/meta_name.csv', encoding='utf-8', header=0, index_col=0)
    set_preffix = list(set(map(lambda column: '^' + (str)(column), ds.loc[ds.index==flag_feature, 'Prefix'].values)))
    set_ordered_preffix = []
    for value in ds.loc[ds.index==flag_feature, 'Prefix'].values:
        patterned_value = '^' + (str)(value)
        if patterned_value in set_preffix and patterned_value not in set_ordered_preffix:
            set_ordered_preffix.append(patterned_value)
    return set_ordered_preffix