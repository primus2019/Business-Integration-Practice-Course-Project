import numpy as np
import pandas as pd
import math


from utils import Preprocess
from utils.Log import printlog
from functools import reduce


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
    printlog('feature_padding_on_hit_rate: mut_exc_feature: {}'.format(mut_exc_feature), printable=False)
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


def feature_woe(ds, features, label_column, features_value=None, encoding='utf-8', header=0, index_col=0):
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    assert not ds.isna().values.any(), 'Temp_support.feature_woe: ds should not contain na data'
    features = [features] if isinstance(features, (str, int)) else features
    features = [ds.columns[f] if isinstance(f, int) else f for f in features]
    label_column = ds.columns[label_column] if isinstance(label_column, int) else label_column
    labels = list(set(np.ravel(ds[[label_column]].values)))
    features_value = [list(set(np.ravel(ds[[column]].values))) for column in features] if not features_value else features_value
    # for fv, column in zip(features_value, features):
    #     for value in fv:
    #         printlog((ds[ds[column] == value][label_column] == labels[0]).sum())
    return [list(map(lambda value, column=column: 
        np.log(
        (((ds[ds[column] == value][label_column] == labels[0]).sum() + 0.5) / (ds[label_column] == labels[0]).sum()) / 
        (((ds[ds[column] == value][label_column] == labels[1]).sum() + 0.5) / (ds[label_column] == labels[1]).sum())),
        fv)) for fv, column in zip(features_value, features)]


def feature_iv(ds, features, label_column, features_woe=None, encoding='utf-8', header=0, index_col=0):
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    features = [features] if isinstance(features, (str, int)) else features
    features = [ds.columns[f] if isinstance(f, int) else f for f in features]
    label_column = ds.columns[label_column] if isinstance(label_column, int) else label_column
    labels = list(set(np.ravel(ds[[label_column]].values)))
    features_value = [list(set(np.ravel(ds[[column]].values))) for column in features]
    # if not features_woe:
    #     features_woe = feature_woe(ds, features, label_column, features_value=features_value, encoding=encoding, header=header, index_col=index_col)
    
    return [reduce(lambda total, value, column=column:
        total + 
        (((ds[ds[column] == value][label_column] == labels[0]).sum() / (ds[label_column] == labels[0]).sum()) - 
        ((ds[ds[column] == value][label_column] == labels[1]).sum()  / (ds[label_column] == labels[1]).sum())) * 
        np.log(
        (((ds[ds[column] == value][label_column] == labels[0]).sum() + 0.5) / (ds[label_column] == labels[0]).sum()) / 
        (((ds[ds[column] == value][label_column] == labels[1]).sum() + 0.5) / (ds[label_column] == labels[1]).sum())
        ),fv,
        0) for fv, column in zip(features_value, features)]
    
    
def select_feature_iv(ds, features, label_column, strict_upper_bound, strict_lower_bound, to_file=None, encoding='utf-8', header=0, index_col=0):
    assert strict_upper_bound > strict_lower_bound, 'Temp_support.select_feature_iv: strict_upper_bound should be larger than strict_lowr_bound'
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    features = [features] if isinstance(features, (str, int)) else features
    features = [ds.columns[f] if isinstance(f, int) else f for f in features]
    printlog('Temp_support.select_feature_iv: calculating feature iv...')
    features_iv = feature_iv(ds, features, label_column, encoding=encoding, header=header, index_col=index_col)
    if to_file:
        printlog('Temp_support.select_feature_iv: saving to path {}...'.format(to_file))
        pd.DataFrame(features_iv, index=features, columns=['iv']).to_csv(to_file, encoding=encoding)
    printlog('Temp_support.select_feature_iv: temporary feature iv: {}'.format([(feature, iv) for feature, iv in zip(features, features_iv)]), printable=False)
    return [feature for feature, iv in zip(features, features_iv) if iv > strict_lower_bound and iv < strict_upper_bound]


def cut(ds, features, threshold=10, bin=10, method='equal-distance', save_path=None, encoding='utf-8', header=0, index_col=0):
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    features = [features] if isinstance(features, (str, int)) else features
    features = [ds.columns[f] if isinstance(f, int) else f for f in features]
    assert not ds.loc[:, features].isna().values.any(), 'Temp_support.cut: ds should not contain na data'
    features = [feature for feature in features if len(list(set(np.ravel(ds[[feature]].values)))) > threshold]
    for feature in features:
        printlog('Temp_support.cut: cutting {}'.format(feature), printable=False)
        if method == 'equal-distance':
            ds.loc[:, feature] = pd.cut(ds[feature], bin)
        elif method == 'equal-frequency':
            ds.loc[:, feature] = pd.qcut(ds[feature], bin, duplicates='drop')
    if save_path:
        ds.to_csv(save_path, encoding=encoding)