import numpy as np
import pandas as pd


def hit_rate(ds, features, threshold, encoding='utf-8', header=0, index_col=0):
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    features = [features] if isinstance(features, str) else features
    return ds.columns[list(filter(lambda column: ds[[column]].notna().values.sum() > threshold, ds.columns))]


def hit_positive_rate(ds, features, label_column, threshold, positive=1, encoding='utf-8', header=0, index_col=0):
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    features = [features] if isinstance(features, str) else features
    label_column = ds.columns[label_column] if isinstance(label_column, int) else label_column
    return list(filter(lambda column: (ds.loc[ds[column].notna(), label_column] == positive).sum() / 
        (ds[column].notna().values.sum() if ds[column].notna().values.sum() != 0 else 1) > threshold, features))
    

def hit_rate(ds, features, threshold, encoding='utf-8', header=0, index_col=0):
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    features = [features] if isinstance(features, str) else features
    return list(filter(lambda column: ds[column].notna().sum() / ds.index.sum() > threshold, ds.columns))