import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


from utils.Log import printlog


def feature_EDA(ds, features, label_column=None, encoding='utf-8', header=0, index_col=0):
    '''
    # Params:

    ds: str/pd.DataFrame, dataset

    features: (list of )feature str

    '''
    assert isinstance(features, (str, list, np.array, pd.Series)), 'EDA.feature_EDA: features should be str, list, np.array or pd.Series; input in {}'.format(type(features))
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    features = [features] if isinstance(features, str) else features
    if label_column:
        label_column = ds.columns[label_column] if isinstance(label_column, int) else label_column
    for feature in features:
        printlog('feature {} has values {} of dtypes {}, distribution {}, label distribution {}'.format(
            feature, 
            list(set(np.ravel(ds[ds[feature].notna()][feature].values))),
            list(set(np.ravel(ds[feature].values.dtype))),
            list(ds[ds[feature].notna()][feature].value_counts().values),
            list(ds[ds[feature].notna()][label_column].value_counts().values) if label_column else '(label_column not given)'
        ))


def feature_na(ds, features, encoding='utf-8', header=0, index_col=0):
    '''
    # Params:

    ds: str/pd.DataFrame, dataset

    features: (list of )feature str

    # Instructions:

    Show na data numbers in features.

    '''
    assert isinstance(features, (str, list, np.array, pd.Series)), 'EDA.feature_na: unexpected features: {}'.format(type(features))
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    features = [features] if isinstance(features, str) else features
    for feature in features:
        printlog('feature {} has {}/{} na sample(s)'.format(
            feature, 
            ds[feature].isnull().sum(), 
            ds.index.size
        ))
