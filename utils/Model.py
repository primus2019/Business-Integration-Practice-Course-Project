from sklearn import tree
import pandas as pd
import numpy as np
import re


from utils import Preprocess
from utils.Log import printlog


def tree_classifier(ds, features, label_column, export_path=None, 
fill_na=None, fill_cat=None, encoding='utf-8', header=0, 
index_col=0, informative=True):
    printlog('Model.tree_classifier: started.', printable=informative)
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    features = [features] if isinstance(features, str) else features
    label_column = ds.columns[label_column] if isinstance(label_column, int) else label_column
    assert fill_na or ds.loc[:, features].isna().sum().sum() == 0, 'Model.tree_classifier: features contains na data; fill_na must be given'
    assert fill_cat or np.dtype('O') not in list(map(lambda column: ds[column].dtype, features)), 'Model.tree_classifier: features contains categorical data; fill_cat must be given'
    if fill_na:
        ds = fill_na(ds, features)
    if fill_cat:
        ds, encoder, features = fill_cat(ds, features)
    # print('features after fill_cat: {}'.format(features))
    clt = tree.DecisionTreeClassifier()
    # print('ds[label_column]: {}'.format(ds.loc[:, label_column].head()))
    # print(ds.head())
    clt = clt.fit(ds.loc[:, features], ds.loc[:, label_column])
    if export_path:
        assert re.search('.dot', export_path), 'Model.tree_classifier: export_path should be in dot format'
        tree.export_graphviz(clt, export_path, feature_names=features)
    else:
        printlog(tree.export_graphviz(clt))
    printlog('Model.tree_classifier: finished.', printable=informative)
    if not fill_cat:
        return clt
    elif fill_cat:
        return clt, encoder, features
