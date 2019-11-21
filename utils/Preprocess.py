import pandas as pd
import numpy as np
import random
import re
import sklearn.preprocessing


from utils.Log import printlog
from functools import reduce


def transpose(ds_path, ds_t_path=False, largeset=False, encoding='utf-8', header=0, index_col=0):
    '''
    Params:

    ds_path: str, dataset path

    ds_t_path(default False): boolean, transposed dataset path
    (should be different from ds_path)

    largeset(default False): boolean, whether to apply low-memory method

    encoding(default 'utf-8'): str, encoding of dataset

    header(default 0): int/list of int, works on pandas.read_csv()
    (learn more at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    index_col(default 0): int/list of int, works on pandas.read_csv()
    (learn more at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    Return: 

    pd.Dataframe, newly transposed dataset

    '''
    assert ds_path != ds_t_path, 'You replace original dataset with processed one(s); assign processed one(s) to new position(s), or delete old dataset with delete.'
    if not largeset:
        if not ds_t_path:
            return pd.read_csv(ds_path, encoding=encoding, header=header, index_col=index_col).T
        else:
            pd.read_csv(ds_path, encoding=encoding, header=header, index_col=index_col).T.to_csv(ds_t_path, encoding=encoding)
            return pd.read_csv(ds_t_path, encoding=encoding, header=header, index_col=index_col)
    elif largeset:
        pass


def split(ds_path, ds_split_path, chunksize=None, fraction=None, shuffle=True, largeset=False, encoding='utf-8', header=0, index_col=0):
    '''
    Params:

    ds_path: str, dataset path

    ds_split_path: str/list of str, split dataset path(s)
    (should be different from ds_path)

    chunksize(default None): int, batch for split dataset(s)
    (strictly one of chunksize and fraction should be valid)

    fraction(default None): float, proportion for split dataset(s)
    (strictly one of chunksize and fraction should be valid)

    shuffle(default True): boolean, whether to randomly shuffle samples when splitting

    largeset(default False): boolean, whether to apply low-memory method for splitting

    encoding(default 'utf-8'): str, encoding of dataset

    header(default 0): int/list of int, works on pandas.read_csv()
    (learn more at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    index_col(default 0): int/list of int, works on pandas.read_csv()
    (learn more at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    '''
    assert ds_path != ds_split_path, 'You replace original dataset with processed one(s); assign processed one(s) to new position(s), or delete old dataset with delete.'
    assert not (chunksize is not None and fraction is not None), 'One and only one of chunksize and fraction should be valid; both valid now.'
    assert chunksize or fraction, 'Only and only one of chunksize and fraction should be valid; both invalid now.'
    if fraction:
        assert fraction <= 1, 'Fraction is over 1; you may mistake it for chunksize, or else you can change it to 1.'
    if chunksize:
        assert chunksize > 1, 'Chunksize is below 1; you may mistake it for faction.'

    if not largeset:
        ds_raw = pd.read_csv(ds_path, encoding=encoding, header=header, index_col=index_col)
        if type(ds_split_path) is not list:
            if chunksize and shuffle:
                ds_raw.sample(n=chunksize).to_csv(ds_split_path, encoding=encoding)
            elif fraction and shuffle:
                ds_raw.sample(frac=fraction).to_csv(ds_split_path, encoding=encoding)
            elif chunksize and not shuffle:
                ds_raw.iloc[:chunksize, :].to_csv(ds_split_path, encoding=encoding)
            elif fraction and not shuffle:
                ds_raw.iloc[:round(ds_raw.shape[0] * fraction), :].to_csv(ds_split_path, encoding=encoding)
            else:
                raise 'Split function meets unworking input.'
        elif type(ds_split_path) is list:
            if chunksize:
                if shuffle:
                    ds_raw = ds_raw.sample(frac=1)
                total_size, remain_size = ds_raw.shape[0], ds_raw.shape[0]
                chunk_cnt = 0
                while remain_size > chunksize:
                    assert chunk_cnt <= len(ds_split_path), 'There is not enough name for split datasets.'
                    ds_raw.iloc[(total_size - remain_size):(total_size - remain_size + chunksize), :].to_csv(ds_split_path[chunk_cnt], encoding=encoding)
                    chunk_cnt += 1
                    remain_size -= chunksize
                assert chunk_cnt <= len(ds_split_path), 'There is not enough name for split datasets.'
                ds_raw.iloc[(total_size - remain_size):total_size, :].to_csv(ds_split_path[chunk_cnt], encoding=encoding)
            elif fraction:
                if shuffle:
                    ds_raw = ds_raw.sample(frac=1)
                total_size = ds_raw.shape[0]
                total_fraction, remain_fraction = 1.0, 1.0
                chunk_cnt = 0
                while remain_fraction > fraction:
                    assert chunk_cnt < len(ds_split_path), 'There is not enough name for split datasets.'
                    ds_raw.iloc[(total_size - round(remain_fraction * total_size)):(total_size - round((remain_fraction - fraction) * total_size)), :].to_csv(ds_split_path[chunk_cnt], encoding=encoding)
                    chunk_cnt += 1
                    remain_fraction -= fraction
                assert chunk_cnt < len(ds_split_path), 'There is not enough name for split datasets.'
                ds_raw.iloc[(total_size - round(remain_fraction * total_size)):total_size, :].to_csv(ds_split_path[chunk_cnt], encoding=encoding)
            else:
                raise 'Split function meets unworking input.'
        else:
            raise 'Split function meets unworking input.'
    elif largeset:
        if type(ds_split_path) is not list:
            if chunksize:
                ds_raw = pd.read_csv(ds_path, nrows=chunksize, encoding=encoding, header=header, index_col=index_col)
                if shuffle:
                    ds_raw = ds_raw.sample(frac=1)
                ds_raw.to_csv(ds_split_path, encoding=encoding)
            elif fraction:
                total_size = 0
                with open(ds_path, encoding=encoding) as file:
                    for line in file:
                        total_size += 1
                total_size -= 1
                ds_raw = pd.read_csv(ds_path, nrows=round(total_size * fraction), encoding=encoding, header=header, index_col=index_col)
                if shuffle:
                    ds_raw = ds_raw.sample(frac=1)
                ds_raw.to_csv(ds_split_path, encoding=encoding)
            else:
                raise 'Split function meets unworking input.'
        elif type(ds_split_path) is list:
            if chunksize:
                ds_itr = pd.read_csv(ds_path, chunksize=chunksize, encoding=encoding, header=header, index_col=index_col)
                for i, ds_item in enumerate(ds_itr):
                    assert i < len(ds_split_path), 'There is not enough name for split datasets.'
                    if shuffle:
                        ds_item = ds_item.sample(frac=1)
                    ds_item.to_csv(ds_split_path[i], encoding=encoding)
            elif fraction:
                total_size = 0
                with open(ds_path, encoding=encoding) as file:
                    for line in file:
                        total_size += 1
                total_size -= 1
                ds_itr = pd.read_csv(ds_path, chunksize=round(total_size * fraction), encoding=encoding, header=header, index_col=index_col)
                for i, ds_item in enumerate(ds_itr):
                    assert i < len(ds_split_path), 'There is not enough name for split datasets.'
                    if shuffle:
                        ds_item = ds_item.sample(frac=1)
                    ds_item.to_csv(ds_split_path[i], encoding=encoding)
            else:
                raise 'Split function meets unworking input.'
        else:
            raise 'Split function meets unworking input.'
    else:
        raise 'Split function meets unworking input.'


def sort(ds_path, save_path, sortby, ascending=True, na_position='last', largeset=False, encoding='utf-8', header=0, index_col=0):
    '''
    # Params:

    ds: str/pd.Dataframe, dataset or dataset path

    save_graph: str, path for csv format file

    sortby: str/list/np.array/pd.Series, feature for sorting

    ascending(default True): boolean, sorting sequence

    na_position(default 'last'): str, either \'last\' or \'first\', position of na in sorting

    encoding(default 'utf-8'): str, encoding of dataset

    header(default 0): int, works on pandas.read_csv()
    (learn more at:https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    index_col(default 0): int, works in pandas.read_csv()
    (learn more at:https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    # Instructions:

    Sort the ds by feature in sortby.

    '''
    assert ds_path != save_path, 'Preprocess.sort: ds_path should be different from save_path'
    assert na_position in ['first', 'last'], 'Preprocess.sort: na_position should be either \'first\' or \'last\'; input is {}'.format(na_position)
    assert re.search('.csv', save_path), 'Preprocess.sort: save_path should match .csv format; add .csv at the suffix'
    assert isinstance(sortby, (str, list, np.array, pd.Series)), 'Preprocess.sort: sortby should be str, list, np.array or pd.Series; input is {}'.format(type(sortby))
    if not largeset:
        ds = pd.read_csv(ds_path, encoding=encoding, header=header, index_col=index_col)
        sortby = [sortby] if isinstance(sortby, str) else sortby
        for onesort in sortby:
            assert onesort in ds.columns, 'Preprocess.sort: sortby is not contained in dataset columns'
        ds.sort_values(by=sortby).to_csv(save_path, encoding=encoding)
    elif largeset:
        pass


def split_train_test_set(ds_path, save_path=None, train_rate=0.9, shuffle=True, largeset=False, encoding='utf-8', header=0, index_col=0):
    '''
    Params:

    ds: str/pd.Dataframe, dataset or dataset path

    file_path(default None): str, if not None, result of checking is saved at the path

    save_graph(default True): boolean, whether save result as graph or csv

    features(default None): list of str/np.array/pd.Series, if not None, only the corresponding features will be checked

    encoding(default 'utf-8'): str, encoding of dataset

    header(default 0): int, works on pandas.read_csv()
    (learn more at:https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    index_col(default 0): int, works in pandas.read_csv()
    (learn more at:https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    '''
    if not largeset:
        ds_raw = pd.read_csv(ds_path, encoding=encoding, header=header, index_col=index_col)
        ds_size = ds_raw.shape[0]
        if shuffle:
            ds_raw = ds_raw.sample(frac=1)
        feature_train = ds_raw.iloc[0: round(ds_size * train_rate), :-1]
        label_train   = pd.DataFrame(ds_raw.iloc[0: round(ds_size * train_rate), -1], columns=[ds_raw.columns[-1]])
        feature_test  = ds_raw.iloc[round(ds_size * train_rate): ds_size, :-1]
        label_test    = pd.DataFrame(ds_raw.iloc[round(ds_size * train_rate): ds_size, -1], columns=[ds_raw.columns[-1]])
        if save_path:
            assert type(save_path) == list, 'Preprocess.split_train_test_set: save_path should be list of str'
            assert len(save_path) == 4, 'Preprocess.split_train_test_set: save_path should be list of 4'
            for path in save_path:
                assert type(path) == str, 'Preprocess.split_train_test_set: save_path should be list of str'
            feature_train.to_csv(save_path[0], encoding=encoding)
            label_train.to_csv(save_path[1], encoding=encoding)
            feature_test.to_csv(save_path[2], encoding=encoding)
            label_test.to_csv(save_path[3], encoding=encoding)
        return feature_train, label_train, feature_test, label_test
    elif largeset:
        pass


def split_measure(label_train, label_test, labels, encoding='utf-8', header=0, index_col=0):
    '''
    # Params:

    label_train: str/pd.DataFrame, trainset label

    label_testL str/pd.DataFrame, testset label

    # Instructions: 

    Check the distribution of labels between trainset and testset.

    '''
    assert type(label_train) in [str, pd.DataFrame], 'Preprocess.split_measure: input should be str or pd.Dataframe'
    assert type(label_test)  in [str, pd.DataFrame], 'Preprocess.split_measure: input should be str or pd.Dataframe'
    if type(label_train) == str:
        label_train = pd.read_csv(label_train, encoding=encoding, header=header, index_col=index_col)
    if type(label_test)  == str:
        label_test  = pd.read_csv(label_test,  encoding=encoding, header=header, index_col=index_col)
    train_size = label_train.shape[0]
    test_size  = label_test.shape[0]
    printlog('Trainset: ')
    for label in labels:
        printlog('\tlabel {}: {}'.format(label, round((label_train == label).values.sum() / train_size, 3)))
    printlog('Testset: ')
    for label in labels:
        printlog('\tlabel {}: {}'.format(label, round((label_test == label).values.sum() /  test_size, 3)))


def fill_na(ds, features, replacement=-99, flag_feature=None, flag_replacement=None, save_path=None, 
    largeset=False, encoding='utf-8', header=0, index_col=0):
    '''
    # Params:

    ds: str/pd.DataFrame, dataset

    features: (list of )str, checked features

    replacement(default -99): int, replacement for na data

    flag_feature(default None): str, possible flag feature for alternative replacement

    flag_replacement(default None): int, alternative replacement for possible flag feature

    # Returns:

    pd.DataFrame of ds

    # Instructions:

    Replace na data of given features, replace with flag_replacement if flag_feature is 1.

    '''
    if not largeset:
        ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
        features = [features] if isinstance(features, str) else features
        for feature in features:
            if flag_feature and flag_replacement:
                printlog('fill na: feature: {}; \tflag_feature: {}'.format(feature, flag_feature), printable=False)
                flag_feature = ds.columns[flag_feature] if isinstance(flag_feature, int) else flag_feature
                ds.loc[(ds[feature].isna()) & (ds[flag_feature] == 1), feature] = flag_replacement
                ds.loc[(ds[feature].isna()) & (ds[flag_feature] == 0), feature] = replacement
            elif (not flag_feature) or (not flag_replacement):
                ds.loc[(ds[feature].isna()), feature] = replacement
            if ds[feature].isna().any():
                raise Exception('still na')
        if save_path:
            ds.to_csv(save_path, encoding=encoding)
        return ds
    elif largeset:
        pass


def fill_cat(ds, features, method='label_encoder', save_path=None, largeset=False, encoding='utf-8', header=0, index_col=0):
    '''
    # Introductions: 

    Automatically check given features and encode categorical columns in pd.DataFrame into numerical-encoded or one-hot-encoded columns.

    Learn more at: label_encoder https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
     and label_binarizer https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html
    
    For label binarizer, columns with categorical data are removed and new columns are generated by \'[old feature name]_[categorical value]\'

    # Params:

    ds: str/pd.DataFrame, (path of )dataset

    features: (list of )features

    method(default 'label_encoder'): 'label_encoder' or 'label_binarizer'

    save_path(default None): str, path of encoded dataset, optional for label encoder while required for label binarizer

    # Returns: 

    Encoded dataset in pd.DataFrame(the newly-generated columns are not inserted in the middle for label binarizer)

    sklearn.preprocessing.LabelEncoder()/sklearn.preprocessing.LabelBinarizer(), with attribute encoder.classes_ as the sequence of the encoding

    List of new features(the newly-generated features are not inserted in the middle for label binarizer)

    '''
    if not largeset:
        ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
        features = [features] if isinstance(features, str) else features
        encoder = []
        if method == 'label_encoder':
            encoder = sklearn.preprocessing.LabelEncoder()
            for feature in features:
                if ds[feature].dtype == np.dtype('O'):
                    encoder.fit(list(set(np.ravel(ds[feature].astype(np.dtype('O')).values))))
                    # encoder.classes_ = list(set(np.ravel(ds[feature].astype(np.dtype(str)).values)))
                    # print(encoder.classes_)
                    ds.loc[:, feature] = encoder.transform(ds[feature].astype(np.dtype(str)))
                    # print(list(set(np.ravel(ds[feature].values))))
                    if ds[feature].dtype == np.dtype('O'):
                        raise Exception('still categorical')
        elif method == 'label_binarizer':
            assert save_path, 'Preprocess.fill_cat: method \'label_binarizer\' split categorical feature into one-hot features, therefore new ds must be saved'
            encoder = sklearn.preprocessing.LabelBinarizer()
            for feature in features:
                if ds[feature].dtype == np.dtype('O'):
                    feature_suffix = list(set(np.ravel(ds[feature].astype(str).values)))
                    tmp_new_feature = [feature + '_' + suffix for suffix in feature_suffix]
                    features.extend(tmp_new_feature)
                    encoder.fit(feature_suffix)
                    # print(encoder.classes_)
                    tmp_new_ds = pd.DataFrame(encoder.transform(ds[feature].astype(np.dtype(str))), columns=tmp_new_feature, index=ds.index)
                    ds = pd.concat([ds, tmp_new_ds], axis=1)
                    del ds[feature]
                    features.remove(feature)
                    if feature in ds.columns:
                        raise Exception('still categorical')
        if save_path:
            ds.to_csv(save_path, encoding=encoding)
        # print(ds.loc[:, features].head())
        return ds, encoder, features
    elif largeset:
        pass


def pattern_to_feature(ds, patterns, encoding='utf-8', header=0, index_col=0):
    '''
    # Params:

    ds: str/pd.DataFrame, dataset

    patterns: regrex format list of str

    # Returns:

    List of features that fit the given regrex in patterns

    '''
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    return [list(filter(lambda column, p=pattern: re.match(p, column), ds.columns))
                for pattern in (patterns if isinstance(patterns, list) else [patterns])]


def clean_outlier(ds, features, measure='std', threshold=3, method='set_na', save_path=None, encoding='utf-8', header=0, index_col=0):
    '''
    # Params:

    ds: str/pd.DataFrame, dataset

    features: list of feature str

    measure(default 'std'): 'std', measurement for cleaning

    threshold(default 3): int

    method(default 'set_na'): 'set_na', 'delete_sample' or 'feature_average'

    '''
    assert method in ['set_na', 'delete_sample', 'feature_average'], 'Preprocess.clean_outlier: method should be \'set_na\', \'delete_sample\' or \'feature_average\''
    assert measure in ['std'], 'Preprocess.clean_outlier: measure should be \'std\''
    assert isinstance(features, (str, list, np.array, pd.Series)), 'Preprocess.clean_outlier: unexpected features type: {}'.format(type(features))
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    features = [features] if isinstance(features, str) else features
    for feature in features:
        # print('working on {}'.format(feature))
        if measure == 'std' and method == 'set_na':
            ds.loc[np.abs(ds[feature] - ds[feature].mean()) > threshold * ds[feature].std(), feature] = np.nan
        elif measure == 'std' and method == 'delete_sample':
            del ds.loc[np.abs(ds[feature] - ds[feature].mean()) > threshold * ds[feature].std(), :]
        elif measure == 'std' and method == 'feature_average':
            ds.loc[np.abs(ds[feature] - ds[feature].mean()) > threshold * ds[feature].std(), feature] = ds.loc[:, feature].mean()
    if save_path:
        ds.to_csv(save_path, encoding=encoding)


def special_feature(ds, features, encoding='utf-8', header=0, index_col=0):
    '''
    # Params:
    
    ds: str/pd.DataFrame, dataset

    features: list of feature str

    # Returns:

    list of features of dtype not in [np.dtype('float64'), np.dtype('int64')].

    '''
    assert isinstance(features, (str, list, np.array, pd.Series)), 'Preprocess.special_feature: unexpected features type: {}'.format(type(features))
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    features = [features] if isinstance(features, str) else features
    special_list = [list(map(lambda x: x not in [np.dtype('float64'), np.dtype('int64')], 
                    list(set(np.ravel(ds[feature].values.dtype)))))[0] 
                    for feature in features]
    special_feature = pd.Series(features)[special_list].values
    return special_feature


def clean_feature(ds, features, pattern=False, method='delete_feature', save_path=None, encoding='utf-8', header=0, index_col=0):
    '''
    # Params:
    
    ds: str/pd.DataFrame, dataset

    features: list of feature str

    method(default 'delete_feature'): 'delete_feature'

    '''
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    features = [features] if isinstance(features, str) else features
    if pattern:
        tmp = [list(filter(lambda column: re.match(pattern, column), ds.columns))
                for pattern in features]
        features=[]
        for t in tmp:
            features.append(t)
    for feature in features:
        if feature:
            if method == 'delete_feature':
                del ds[feature]
    if save_path:
        ds.to_csv(save_path, encoding=encoding)


# to be tested
def pop_feature(ds, features, save_path=None, pop_path=None, encoding='utf-8', header=0, index_col=0):
    '''
    # Instruction: 

    Abstract and remove features from dataset.

    # Params:

    ds: str/pd.DataFrame

    # Returns:

    pd.DataFrame of given features

    '''
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    features = [features] if isinstance(features, str) else features
    ds_pop = ds[features].copy()
    for feature in features:
        del ds[feature]
    if save_path:
        ds.to_csv(save_path, encoding=encoding)
    if pop_path:
        ds_pop.to_csv(pop_path, encoding=encoding)
    return ds_pop



def clean_poor_sample(ds, threshold, save_path=None, encoding='utf-8', header=0, index_col=0):
    '''
    # Params:

    ds: str/pd.DataFrame, dataset

    threshold: int, threshold for least notna features of sample

    '''
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    ds = ds[ds.notna().sum(axis=1) > threshold]
    if save_path:
        ds.to_csv(save_path, encoding=encoding)


def clean_poor_feature(ds, threshold, save_path=None, encoding='utf-8', header=0, index_col=0):
    '''
    # Params:

    ds: str/pd.DataFrame, dataset

    threshold: int, threshold for least notna samples of feature

    '''
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    ds = ds.loc[:, ds.notna().sum(axis=0) > threshold]
    if save_path:
        ds.to_csv(save_path, encoding=encoding)


def clean_dull_feature(ds, threshold, label_column, save_path=None, encoding='utf-8', header=0, index_col=0):
    '''
    # Params:

    ds: str/pd.DataFrame, dataset

    threshold: int, threshold for most simple pair (feature value, label value) of feature

    label_column: str/int, index or str of label column
    
    '''
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    label_column = ds.columns[label_column] if isinstance(label_column, int) else label_column
    ds = pd.concat([ds.iloc[:, ds.columns != label_column].loc[:, ds.iloc[:, ds.columns != label_column]
        .apply(lambda column: column.astype(str) + '_' + ds.loc[:, label_column].astype(str))
        .apply(lambda column: column.value_counts().max() / column.value_counts().sum())
        <= threshold], ds[label_column]], axis=1)
    if save_path:
        ds.to_csv(save_path, encoding=encoding)

# only for bivariate label
def clean_lowIV_feature(ds, label_column, threshold=0.02, save_path=None, encoding='utf-8', header=0, index_col=0):
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    label_column = ds.columns[label_column] if isinstance(label_column, int) else label_column
    ds_t = ds.loc[:, ds.columns != label_column]
    for i in range(55, 255):
        ct = pd.crosstab(ds_t.iloc[:, i], ds.loc[:, label_column])
        print(ct)
        print(pd.melt(ct))
    # print(np.divide(ct.values, [pd.melt(ct)['value'], pd.melt(ct)['value']]))

    # tl = pd.Series(map(lambda column: reduce(lambda accum, dist: accum + (dist[0] - dist[1]) / (dist[0] + dist[1]) * np.log(dist[0] / dist[1]) 
    #     if dist[1] != 0 else (dist[1] / ds_t[ds_t[column] == dist] - dist[0]) * np.log(dist[1] / dist[0]), 
    #     pd.crosstab(ds_t.loc[:, column], ds.loc[:, label_column]).values, 0), ds_t.columns), index=ds_t.columns)
    # print(tl)

    # tl = pd.Series(map(lambda column: reduce(lambda accum, dist: accum + (dist[0] - dist[1]) / (dist[0] + dist[1]) * np.log(dist[0] / dist[1]) 
    #     if dist[1] != 0 else (dist[1] - dist[0]) * np.log(dist[1] / dist[0]), 
    #     pd.crosstab(ds_t.loc[:, column], ds.loc[:, label_column]).values, 0), ds_t.columns), index=ds_t.columns)
    # print('Apart from label_column {}, totally {}/{} features have IV > {}'.format(
    #     label_column,
    #     tl.size,
    #     ds.columns.size - 1,
    #     threshold
    # ))
    # ds = pd.concat([ds_t[(tl > threshold).index], ds[[label_column]]], axis=1)
    # if save_path:
    #     ds.to_csv(save_path, encoding=encoding)
    