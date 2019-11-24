import numpy as np
import pandas as pd
import math


from utils import Preprocess
from utils.Log import printlog


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

    
def feature_padding_on_hit_rate(ds, features, preffix_patterns, encoding='utf-8', header=0, index_col=0):
    ## get suffix of features in given class
    classed_class_features = Preprocess.pattern_to_feature(ds, preffix_patterns, encoding=encoding)
    ds = pd.read_csv(ds, encoding='gb18030', header=header, index_col=index_col) if isinstance(ds, str) else ds
    ## tmp: class suffix inflattened
    tmp = [list(map(lambda fc, pf=preffix: fc[len(pf) - 1:], feature_class)) for preffix, feature_class in zip(preffix_patterns, classed_class_features)]
    class_suffix = []
    for t in tmp:
        class_suffix.extend(t)
    ## class_suffix: class suffix unique flattened
    class_suffix = list(set(class_suffix))
    # print('feature_padding: preffix_patterns = {}'.format(preffix_patterns))
    ## get features with mutually exclusive suffixs
    mut_exc_feature = []
    for suffix in class_suffix:
        tmp_hit_rate = 0
        tmp_output_feature = ''
        for i, t in enumerate(tmp):
            if suffix in t:
                tmp_feature = preffix_patterns[i][1:] + suffix
                tmp_feature_hit_rate = ds[tmp_feature].notna().sum() / ds.shape[0]
                if tmp_feature_hit_rate > tmp_hit_rate:
                    tmp_hit_rate = tmp_feature_hit_rate
                    tmp_output_feature = tmp_feature
        if tmp_output_feature != '':
            mut_exc_feature.append(tmp_output_feature)
    printlog('feature_padding_on_hit_rate: mut_exc_feature: {}'.format(mut_exc_feature))
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


def two_layer_tree(ds, feature_1, feature_2, label_column, to_file=None, printable=False, encoding='utf-8', header=0, index_col=0):
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    feature_1 = ds.columns[feature_1] if isinstance(feature_1, int) else feature_1
    feature_2 = ds.columns[feature_2] if isinstance(feature_2, int) else feature_2
    label_column = ds.columns[label_column] if isinstance(label_column, int) else label_column
    value_1 = list(set(np.ravel(ds[ds[feature_1].notna()][feature_1].values)))
    value_2 = list(set(np.ravel(ds[ds[feature_2].notna()][feature_2].values)))
    label   = list(set(np.ravel(ds[ds[label_column].notna()][label_column].values)))
    if printable:
        for v1 in value_1:
            for v2 in value_2:
                printlog('value - {}: {}, {}: {}; dist - {}: {}, {}: {}'.format(
                    value_1,
                    v1,
                    value_2,
                    v2,
                    label[0],
                    ds[(ds[feature_1] == v1) & (ds[feature_2] == v2) & (ds[label_column] == label[0])].shape[0],
                    label[1],
                    ds[(ds[feature_1] == v1) & (ds[feature_2] == v2) & (ds[label_column] == label[1])].shape[0]
                ))
    if to_file:
        with open(to_file, 'w+') as file:
            file.write(
'''digraph Tree {}
node [shape=box] ;
0 [label=\"{}\"] ;
'''.format('{', feature_1))
            for i, v1 in enumerate(value_1):
                file.write('0 -> {} ;\n'.format(i + 1))
                file.write('{} [label=\"{}\\n{}: {}\\n{}: {}\"] ;\n'.format(
                    i + 1, 
                    v1,
                    label[0],
                    ds[(ds[feature_1] == v1) & (ds[label_column] == label[0])].shape[0],
                    label[1],
                    ds[(ds[feature_1] == v1) & (ds[label_column] == label[1])].shape[0]
                ))
                for j, v2 in enumerate(value_2):
                    file.write('{} [label=\"{}\\n{}: {}\\n{}: {}\"] ;\n'.format(
                        len(value_1) + len(value_2) * i + j + 1,
                        v2,
                        label[0],
                        ds[(ds[feature_1] == v1) & (ds[feature_2] == v2) & (ds[label_column] == label[0])].shape[0],
                        label[1],
                        ds[(ds[feature_1] == v1) & (ds[feature_2] == v2) & (ds[label_column] == label[1])].shape[0]
                    ))
                    file.write('{} -> {} ;\n'.format(
                        i + 1,
                        len(value_1) + len(value_2) * i + j + 1
                        # round(2 / math.cos(60 - 120 / (len(value_2) - 1) * j), 3),
                        # 60 - 120 / (len(value_2) - 1) * j
                    ))
            file.write('{} [label=\"layer 1: {}\\nlayer 2: {}\"] ;\n'.format(len(value_1) * len(value_2) + len(value_1) + 1, feature_1, feature_2))
            file.write('}')