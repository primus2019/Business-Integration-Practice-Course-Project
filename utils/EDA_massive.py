'''
ddd
'''
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import re


from utils.Log import printlog


def EDA(ds, data_type, folder=None, save_graph=True, encoding='utf-8', header=0, index_col=0, largeset=False, nrows=1000):
    '''
    # Params:

    ds_path: str/pd.Dataframe , dataset path or dataset

    data_type: str, either 'feature' or 'label'; decides EDA mode

    folder(default None): str, if not None, save EDA files in the folder

    save_graph(default True): boolean, whether save image files

    encoding(default 'utf-8'): str, encoding of dataset

    header(default 0): int/list of int, works on pandas.read_csv()
    (learn more at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    index_col(default 0): int/list of int, works on pandas.read_csv()
    (learn more at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    largeset(default False): boolean, whether to apply low-memory method for EDA

    nrows(default 1000): int, works on pandas.read_csv(), work only when largeset is True
    (learn more at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    '''
    types = ['feature', 'label']
    assert data_type in types, 'Types are not valid; should be \'feature\' or \'label\''

    if type(ds) is str:
        printlog('---------------------EDA of {}---------------------'.format(os.path.basename(ds)))
    if not largeset:
        ds_raw = pd.read_csv(ds, encoding='utf-8', index_col=index_col, header=header) if isinstance(ds, str) else ds
        ## size, head and label
        printlog('SIZE:            [{} sample(row) * {} feature(column)]'.format(ds_raw.shape[0], ds_raw.shape[1]))
        printlog('HEAD:            \n{}'.format(ds_raw.head()))
        if data_type == 'label':
            printlog('LABELS:          {}'.format(list(set(np.ravel(ds_raw.values)))))
        ## na data, feature type
        na_data_path, fe_data_path = None, None
        if folder and save_graph:
            na_data_path = os.path.join(folder, 'record_na_data.png')
            fe_data_path = os.path.join(folder, 'record_feature_type.png')
        if folder and not save_graph:
            na_data_path = os.path.join(folder, 'record_na_data.csv')
            fe_data_path = os.path.join(folder, 'record_feature_type.csv')
        na_data(ds_raw, na_data_path)
        feature_type(ds_raw, fe_data_path)
    elif largeset:
        if type(ds) is str:
            rows, columns = 0, 0
            with open(ds, encoding=encoding) as file:
                for line in file:
                    rows += 1
                    columns = len(line.split(',')) - 1
            rows -= 1
            printlog('[{} sample(row) * {} feature(column)]'.format(rows, columns))
            ds_raw = pd.read_csv(ds, nrows=nrows, encoding=encoding, header=header, index_col=index_col)
        else: 
            ds_raw = ds
        EDA(ds_raw, data_type, encoding=encoding, header=header, index_col=index_col)
    printlog('+++++++++++++++++++++EDA of {}+++++++++++++++++++++'.format(os.path.basename(ds)))


def shape(ds, largeset=False, encoding='utf-8', header=0, index_col=0):
    '''
    # Params:

    ds: pd.Dataframe or str of dataset path
    
    '''
    if type(ds) == str:
        ds = pd.read_csv(ds, encoding=encoding, header=0, index_col=0)
    printlog('SAMPLE(row):           {}'.format(ds.shape[0]))
    printlog('FEATURE/LABEL(column): {}'.format(ds.shape[1]))    


def labels(ds, column, encoding='utf-8', header=0, index_col=0):
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    column = ds.columns[column] if isinstance(column, int) else column
    assert column in ds.columns, 'EDA_massive.labels: column is not contained in ds'
    return list(set(np.ravel(ds[[column]].values)))



def sparse_feature(ds, features=None, file_path=None, measure='std', threshold=0.01, largeset=False, encoding='utf-8', header=0, index_col=0):
    '''
    # Params: 

    ds: pandas.Dataframe, numpy.ndarray or str of dataset path shaped [n of samples, n of features]

    features(default None): str/list of str, if not None, only corresponding features in ds will be checked

    file_path(default None): str, if not None, result is saved at the path

    measure(default 'std'): str, either 'mean' or 'std', deciding the calculation of feature performance
    (mean threshold are compared with features' absolute means)

    threshold(default 0.01): float, threshold for deciding whether a feature is sparse

    largeset(default Faslse): boolean, whether to apply low-memory method for sparse detection

    encoding(default 'utf-8'): str, encoding of dataset

    header(default 0): int, works on pandas.read_csv()
    (learn more at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    index_col(default 0 ): int, works on pandas.read_csv()
    (learn more at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    # Return:
    
    pandas.Series, boolean values of shape [n of features]
    (true represent the feature is not sparse; False represent sparse)

    '''
    assert measure in ['mean', 'std'], 'EDA.sparse_feature: parameter measure should be either \'mean\' or \'std\', {} is given'.format(measure)
    if type(ds) == str:
        ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col)
    if measure == 'mean':
        insparse_feature = ((ds != 0).abs().mean() > threshold)
    elif measure == 'std':
        insparse_feature = (ds != 0).std() > threshold
    printlog('SPARSE:            {}/{} features(threshold: {}, measure: {})'.format((insparse_feature == True).sum(), insparse_feature.size, threshold, measure))
    if file_path:
        insparse_feature.to_csv(file_path, encoding=encoding, header=True)


def na_data(ds, file_path=None, save_graph=True, features=None, encoding='utf-8', header=0, index_col=0):
    '''
    # Params:

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
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    columns = features if features else ds.columns
    series_na = pd.Series()
    for feature in columns:
        series_na = series_na.append(pd.Series([ds[feature].isna().sum()]), ignore_index=True)
    printlog('NA:            {}/{} features(totally {} data)'.format((series_na > 0).sum(), series_na.size, series_na.sum()))
    if file_path and save_graph:
        assert re.search(r'.png', file_path) or re.search(r'.jpg', file_path) or re.search(r'.jpeg', file_path), 'EDA.na_data: file_path is not in image format; use .png, .jpg, .jpeg suffix'
        sns.distplot(series_na, kde=False)
        plt.title('Na data in features')
        plt.xlabel('feature count')
        plt.ylabel('Na data count')
        plt.savefig(file_path)
        plt.close()
    if file_path and not save_graph:
        assert re.search(r'.csv', file_path), 'EDA.na_data: file_path does not match tabular format; use .csv suffix'
        series_na.to_csv(file_path, encoding=encoding, header=True)


def outlier_data(ds, file_path=None, features=None, measure='std', threshold=3, encoding='utf-8', header=0, index_col=0):
    '''
    # Params:

    ds: str/pd.Dataframe, dataset or dataset path

    file_path(default None): str, if not None, the result is saved in path

    features(default None): list of str/np.array/pd.Series, if not None, only corresponding features will be checked
    '''
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    series_outlier = pd.Series()
    columns = features if features else ds.columns
    if measure == 'std':
        for feature in ds.columns:
            single_feature_outlier = ds[feature][np.abs(ds[feature] - ds[feature].mean()) > threshold * ds[feature].std()].sum()
            series_outlier = series_outlier.append(pd.Series([single_feature_outlier]), ignore_index=True)
    printlog('OUTLIER:            {}/{} features(threshold: {}, measure: {})'.format((series_outlier > 0).values.sum(), series_outlier.size, threshold, measure))
    if file_path:
        series_outlier.to_csv(file_path, encoding=encoding, header=True)


def feature_type(ds, file_path=None, save_graph=True, encoding='utf-8', header=0, index_col=0):
    '''
    # Params:

    ds: str/pd.DataFrame, dataset

    # Instructions:

    Check dataset feature types

    '''
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    type_count = ds.dtypes.value_counts()
    printlog('FEATURE TYPECOUNT: \n{}'.format(type_count))
    if file_path and save_graph:
        assert re.search(r'.png', file_path) or re.search(r'.jpg', file_path) or re.search(r'.jpeg', file_path), 'EDA.na_data: file_path is not in image format; use .png, .jpg, .jpeg suffix'
        # printlog([(str)(value) for value in type_count.index.values])
        # printlog(type_count.values)
        plt.bar([(str)(value) for value in type_count.index.values], type_count.values)
        plt.title('Feature type')
        plt.xlabel('dtype')
        plt.ylabel('Feature count')
        plt.savefig(file_path)
        plt.close()
    if file_path and not save_graph:
        assert re.search(r'.csv', file_path), 'EDA.na_data: file_path does not match tabular format; use .csv suffix'
        type_count.to_csv(file_path, encoding=encoding, header=header)


def date_feature(ds, feature, labels=None, label_column=None, file_path=None, save_graph=True, encoding='utf-8', header=0, index_col=0):
    '''
    # Params:

    ds: str/pd.Dataframe, dataset or dataset path

    feature: str, feature in datetime format

    labels(default None): list, dataset labels

    label_column(default None): str/int, index of label column in ds or label column in ds; if both labels and label_column is not None, date_feature details will be checked by label

    file_path(default None): str, if not None, result of checking is saved at the path

    save_graph(default True): boolean, whether save result as graph or csv

    features(default None): list of str/np.array/pd.Series, if not None, only the corresponding features will be checked

    encoding(default 'utf-8'): str, encoding of dataset

    header(default 0): int, works on pandas.read_csv()
    (learn more at:https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    index_col(default 0): int, works in pandas.read_csv()
    (learn more at:https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    '''
    assert type(feature) == str, 'EDA_massive: feature should be str; {} is entered'.format(feature)
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    assert feature in ds.columns, 'EDA_massive: feature should be contained in ds columns'
    labels_countby_date = []
    if labels and label_column:
        label_column = ds.columns[label_column] if isinstance(label_column, int) else label_column
        for label in labels:
            ds_tmp = ds[ds[label_column] == label]
            ds_tmp = ds_tmp[feature].astype('datetime64')
            labels_countby_date.append(ds_tmp.groupby([ds_tmp.dt.year, ds_tmp.dt.month]).count())
    ds = ds[feature].astype('datetime64')
    date_count = ds.groupby([ds.dt.year, ds.dt.month]).count()
    printlog('FEATURE {} DATE COUNT: \n{}'.format(feature, date_count))
    if file_path and save_graph:
        assert re.search(r'.png', file_path) or re.search(r'.jpg', file_path) or re.search(r'.jpeg', file_path), 'EDA.na_data: file_path is not in image format; use .png, .jpg, .jpeg suffix'
        plt.figure(figsize=[10, 10])
        if labels and label_column:
            prev = pd.Series(np.zeros(date_count.size), index=date_count.index)
            plt.bar([(str)(value) for value in prev.index.values], prev.values, bottom=None)
            for i, label in enumerate(labels_countby_date):
                if prev is not None:
                    prev = pd.Series([prev[index] for index in label.index], index=label.index)
                # print(prev)
                # print([(str)(value) for value in label.index.values])
                plt.bar([(str)(value) for value in label.index.values], label.values, bottom=prev, label='label: {}'.format(labels[i]))
                i = 0
                for j, index in enumerate(date_count.index):
                    if i == label.index.size:
                        break
                    if index == label.index[i]:
                        # print('index: {}'.format(index))
                        plt.plot(j, label[i] + prev[i], marker='D')
                        plt.text(j - 0.3, label[i] + prev[i] + 1, (str)(label[i]))
                        i += 1
                prev = label
        elif not labels or not label_column:
            plt.bar([(str)(value) for value in date_count.index.values], date_count.values)
        for i, data in enumerate(date_count.values):
            plt.text(i - 0.3, 2, (str)(data))
        plt.title('Count on date of feature {}'.format(feature))
        plt.xlabel('Date range')
        plt.xticks(rotation=90)
        plt.ylabel('Sample number')
        plt.legend()
        plt.savefig(file_path)
        plt.close()
    if file_path and not save_graph:
        assert re.search(r'.csv', file_path), 'EDA.na_data: file_path does not match tabular format; use .csv suffix'
        date_count.to_csv(file_path, encoding=encoding, header=header)


def poor_sample(ds, threshold, encoding='utf-8', header=0, index_col=0):
    '''
    # Params:

    ds: str/pd.Dataframe, dataset or dataset path

    threshold: int, samples are checked by threshold number of notNa features

    encoding(default 'utf-8'): str, encoding of dataset

    header(default 0): int, works on pandas.read_csv()
    (learn more at:https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    index_col(default 0): int, works in pandas.read_csv()
    (learn more at:https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    '''    
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    printlog('poor samples: {}/{} samples contain more than {} notNa features'.format(
        (ds.notna().sum(axis=1) > threshold).sum(),
        ds.index.size,
        threshold
    ))


def poor_feature(ds, threshold, encoding='utf-8', header=0, index_col=0):
    '''
    # Params:

    ds: str/pd.Dataframe, dataset or dataset path

    threshold: int, features are checked by threshold number of samples

    encoding(default 'utf-8'): str, encoding of dataset

    header(default 0): int, works on pandas.read_csv()
    (learn more at:https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    index_col(default 0): int, works in pandas.read_csv()
    (learn more at:https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    '''
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    printlog('poor features: {}/{} features contain more than {} notNa samples'.format(
        (ds.notna().sum() > threshold).sum(),
        ds.columns.size,
        threshold
    ))


def dull_feature(ds, threshold, label_column, encoding='utf-8', header=0, index_col=0):
    '''
    # Params:

    ds: str/pd.Dataframe, dataset or dataset path

    threshold: int, features are checked by threshold number of dull data

    encoding(default 'utf-8'): str, encoding of dataset

    header(default 0): int, works on pandas.read_csv()
    (learn more at:https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    index_col(default 0): int, works in pandas.read_csv()
    (learn more at:https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    # Instructions:

    Discard features that have oversized appearance of single (value, label) pair.

    '''
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    label_column = ds.columns[label_column] if isinstance(label_column, int) else label_column
    # if isinstance(label_column, int):
    #     printlog('dull samples: {}/{} features contain more than {} dull samples'.format(
    #         (ds.iloc[:, ds.columns != ds.columns[label_column]]
    #         .apply(lambda column: column.astype(str) + '_' + ds.iloc[:, label_column].astype(str))
    #         .apply(lambda column: column.value_counts().max() / column.value_counts().sum())
    #         > threshold).sum(),
    #         ds.columns.size - 1,
    #         threshold
    #     ))
    printlog('dull samples: {}/{} features contain more than {} dull samples'.format(
        (ds.iloc[:, ds.columns != label_column]
        .apply(lambda column: column.astype(str) + '_' + ds.loc[:, label_column].astype(str))
        .apply(lambda column: column.value_counts().max() / column.value_counts().sum())
        > threshold).sum(),
        ds.columns.size - 1,
        threshold
    ))

    
