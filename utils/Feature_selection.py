import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.linear_model import Lasso

from utils.Log import printlog

__all__ = [
    'hit_positive_rate',
    'select_on_lasso'
]

def hit_rate(ds, features, threshold, encoding='utf-8', header=0, 
index_col=0, informative=True):
    printlog('Feature_selection.hit_rate: started.', printable=informative)
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    features = [features] if isinstance(features, str) else features
    printlog('Feature_selection.hit_rate: finished.', printable=informative)
    return ds.columns[list(filter(lambda column: ds[[column]].notna().values.sum() > threshold, ds.columns))]


def hit_positive_rate(ds, features, label_column, threshold, 
positive=1, na_replacement=None, encoding='utf-8', header=0, index_col=0, 
informative=True):
    printlog('Feature_selection.hit_positive_rate: started.', printable=informative)
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    features = [features] if isinstance(features, str) else features
    label_column = ds.columns[label_column] if isinstance(label_column, int) else label_column
    printlog('Feature_selection.hit_positive_rate: finished.', printable=informative)
    if na_replacement:
        return list(filter(lambda column: (ds.loc[ds[column] != na_replacement, label_column] == positive).sum() / 
            ((ds[column] != na_replacement).sum() if (ds[column] != na_replacement).sum() != 0 else 1) > threshold, features))
    else:
        return list(filter(lambda column: (ds.loc[ds[column].notna(), label_column] == positive).sum() / 
            (ds[column].notna().values.sum() if ds[column].notna().values.sum() != 0 else 1) > threshold, features))


def hit_rate(ds, features, threshold, encoding='utf-8', header=0, 
index_col=0, informative=True):
    printlog('Feature_selection.hit_rate: started.', printable=informative)
    ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col) if isinstance(ds, str) else ds
    features = [features] if isinstance(features, str) else features
    printlog('Feature_selection.hit_rate: finished.', printable=informative)
    return list(filter(lambda column: ds[column].notna().sum() / ds.shape[0] > threshold, features))


# TODO: type check
def select_on_lasso(
    X,
    y,
    lasso_params=None,
    abs_thresh=None,
    sort_index=None,
    sorted=False,
    encoding='utf-8',
    header=0,
    index_col=0):
    """Select features based on Lasso coefficients, by either absolute value 
    threshold or sorting results.

    Parameters
    ----------
    X : DataFrame of size [n_samples, n_features]
        All features in DataFrame should be checked. 

    y : Series of size [n_samples]
        Labels of X.
    
    lasso_params : dict
        Feature params of Lasso, applied through Lasso.set_params(**lasso_params).

    abs_thresh : int or float
        Numeric threshold of Lasso absolute coefficients in selection.

    sort_index : int or turple of form (int or None, int or None), or list of 
    form (int or None, int or None)
        Index of feature Lasso absolute coefficients array, in ascending manner. If single
        int N is passed, features with largest N Lasso coefficents are selected.
    
    Either abs_thresh or sort_index can work; while both are passed, ValueError raises.
    
    sorted : boolean
        Whether to return best_features and all_features sorted by Lasso coefficents.

    Returns
    -------
    best_features : Series
        Series indexed by selected features, named 'lasso_coef' and contains 
        corresponding feature Lasso coefficients. 
    
    all_features : Series of size [n_features]
        Series indexed by input features, named 'lasso_coef' and contains 
        corresponding feature Lasso coefficients. 

    """
    if abs_thresh and sort_index: 
        raise TypeError('Should not input both abs_thresh and sort_index. ')

    lasso = Lasso().set_params(**lasso_params)
    lasso.fit(X.values, y.values)

    all_features = pd.Series(data=lasso.coef_, index=X.columns, name='lasso_coef')

    if abs_thresh:
        best_features = all_features[np.abs(all_features) > abs_thresh]
    if sort_index:
        if isinstance(sort_index, int):
            best_features = all_features.sort_values().iloc[-sort_index:]
        else:
            best_features = all_features.sort_values().iloc[sort_index[0], sort_index[1]]
    
    if sorted:
        all_features.sort_values(inplace=True)
        best_features.sort_values(inplace=True)
    
    return best_features, all_features


# TODO: type check
def select_on_xgb(
    X,
    y,
    xgb_params,
    abs_thresh=None,
    sort_index=None,
    sorted=False,
    encoding='utf-8',
    header=0,
    index_col=0):
    """Select features based on Xgboost feature importances, by either absolute value 
    threshold or sorting results.

    Parameters
    ----------
    X : DataFrame of size [n_samples, n_features]
        All features in DataFrame should be checked. 

    y : Series of size [n_samples]
        Labels of X.
    
    xgb_params : dict
        Feature params of Xgboost, applied through XgbClassifier.set_params
        (**xgb_params).

    abs_thresh : int or float
        Numeric threshold of Xgboost feature importances in selection.

    sort_index : int or turple of form (int or None, int or None), or list of 
    form (int or None, int or None)
        Index of Xgboost feature importances array, in ascending manner. If single
        int N is passed, features with largest N feature importances are selected.
    
    Either abs_thresh or sort_index can work; while both are passed, ValueError 
        raises.
    
    sorted : boolean
        Whether to return best_features and all_features sorted by Xgboost 
        feature importances

    Returns
    -------
    best_features : Series
        Series indexed by selected features, named 'lasso_coef' and contains 
        corresponding feature Lasso coefficients. 
    
    all_features : Series of size [n_features]
        Series indexed by input features, named 'lasso_coef' and contains 
        corresponding feature Lasso coefficients. 

    """
    if abs_thresh and sort_index: 
        raise TypeError('Should not input both abs_thresh and sort_index. ')

    xgb = XGBClassifier().set_params(**xgb_params)
    xgb.fit(X.values, y.values)

    all_features = pd.Series(
        data=xgb.feature_importances_, index=X.columns, name='lasso_coef')

    if abs_thresh:
        best_features = all_features[np.abs(all_features) > abs_thresh]
    if sort_index:
        if isinstance(sort_index, int):
            best_features = all_features.sort_values().iloc[-sort_index:]
        else:
            best_features = all_features.sort_values().iloc[sort_index[0], sort_index[1]]
    
    if sorted:
        all_features.sort_values(inplace=True)
        best_features.sort_values(inplace=True)
    
    return best_features, all_features
    