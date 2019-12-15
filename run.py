from utils import EDA_massive, Preprocess, Log, EDA, Feature_selection, Model, Temp_support, Assess
from utils.Log import printlog

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Lasso
from xgboost import XGBClassifier
from collections import Counter
import matplotlib.pyplot as plt
from joblib import dump, load
from functools import reduce
from sklearn import tree
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import potplayer
import itertools
import traceback
import winsound
import logging

def run():
    printlog('-----------------------------------start presetting-----------------------------------')
    ## hyperparams
    ## feature selection
    hit_pos_rate_upper = 0.5
    hit_pos_rate_lower = 0.2
    tree_max_depth = 2
    iv_upper_thresh = 999
    iv_lower_thresh = 0.2
    lasso_alpha = 1.0
    lasso_coef = 1e-05
    ## model
    xgb_FP_grad_mul = 0.3
    xgb_FN_grad_mul = 1.2
    xgb_zero_proba_cutoff = 0.5
    ## settings
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'SimHei'
    Log.clear_log(creative=True)
    ##
    ds_path             = 'data/data.csv'               # raw dataset
    ds_merged           = 'data/ds_merged.csv'          # raw dataset merged with population dataset
    ds_na               = 'tmp/ds_na.csv'               # merged dataset clear of na data
    ds_cat              = 'tmp/ds_cat.csv'              # merged dataset clear of categorical feature
    ds_cut              = 'tmp/ds_cut.csv'              # merged dataset cut for IV feature selection
    ds_varied           = 'tmp/ds_varied.csv'           # merged dataset varied
    ds_train            = 'tmp/ds_train.csv'            # split train dataset
    ds_valid            = 'tmp/ds_valid.csv'            # split validation dataset
    ds_test             = 'tmp/ds_test.csv'             # split test dataset
    iv_detail           = 'iv/iv_detail.csv'            # dataset with feature IVs
    lasso_detail        = 'lasso/lasso_detail.csv'      # dataset with feature lasso coefficients
    xgb_detail          = 'xgb/xgb_detail.csv'          # dataset with feature xgb importances
    fe_iv               = 'features/fe_iv.csv'          # selected feature by IV
    fe_lasso            = 'features/fe_lasso.csv'       # selected feature by lasso coefficients
    fe_xgb              = 'features/fe_xgb.csv'         # selected feature by xgb importances
    tree_gate           = 'tmp/tree_gate.joblib'        # trained tree model
    model_xgb           = 'xgb/model_xgb.joblib'        # trained xgb model
    model_xgb_optim     = 'xgb/model_xgb_optim.joblib'        # trained xgb model optimized
    plot_gate_tree      = 'tmp/gate_tree.dot'           # plot of tree model
    fe_gate_hit         = 'features/fe_gate_hit.csv'    # selected gate feature
    fe_gate_tree        = 'features/fe_gate_tree.csv'   # selected tree feature
    ## class 1, 2, 4 variables
    fe_gate_pattern     = ['^sl_', '^fr_', '^alu_']
    ## class 3, 5, 6, 7, 8 variables
    fe_model_pattern    = ['^ir_', '^als_', '^cf_', '^cons_', '^pd_']

    printlog('-----------------------------------feature preprocess-----------------------------------')
    printlog('-----------------------------------prepare dataset-----------------------------------')
    Preprocess.fill_na(ds_merged, 'all', replacement=-1, save_path=ds_na, encoding='gb18030')
    Preprocess.fill_cat(ds_na, 'all', save_path=ds_cat, encoding='gb18030')
    varyDataset(ds=ds_cat, save_path=ds_varied)
    generateExperienceFeature(ds_varied)
    train_fe, valid_fe, test_fe, train_lb, valid_lb, test_lb = Preprocess.train_validation_test_split(ds_varied, -1, 0.8, 0.1, 0.1, encoding='gb18030')
    printlog('train label proportion:      {}; '.format(train_lb.sum() / train_lb.count()))
    printlog('validation label proportion: {}'.format(valid_lb.sum() / valid_lb.count()))
    printlog('test label proportion:       {}'.format(test_lb.sum() / test_lb.count()))
    printlog('train feature shape:         {}; '.format(train_fe.shape))
    printlog('validation feature shape:    {}; '.format(valid_fe.shape))
    printlog('test feature shape:          {}; '.format(test_fe.shape))
    pd.concat([train_fe, train_lb], axis=1, sort=True).to_csv(ds_train, encoding='gb18030')
    pd.concat([valid_fe, valid_lb], axis=1, sort=True).to_csv(ds_valid, encoding='gb18030')
    pd.concat([test_fe,  test_lb],  axis=1, sort=True).to_csv(ds_test,  encoding='gb18030')

    printlog('-----------------------------------feature selection-----------------------------------')
    printlog('-----------------------------------feature selection on gate feature and tree classifier-----------------------------------')
    fe_gate       = refreshModelFeature(ds_train, fe_gate_pattern)
    ## gate feature
    fe_gate_upper = Feature_selection.hit_positive_rate(ds_train, fe_gate, -1, hit_pos_rate_upper, na_replacement=-1, encoding='gb18030')
    fe_gate_lower = Feature_selection.hit_positive_rate(ds_train, fe_gate, -1, hit_pos_rate_lower, na_replacement=-1, encoding='gb18030')
    Log.itersave(fe_gate_hit, fe_gate_upper)
    Log.itersave(fe_gate_tree, [fe for fe in fe_gate_lower if fe not in fe_gate_upper])
    ## tree model
    tcl = Model.tree_classifier(ds=ds_train, 
        features=Log.iterread(fe_gate_tree), label_column=-1, 
        max_depth=tree_max_depth, encoding='gb18030', export_path=plot_gate_tree) ## only if fill_cat apply method='label_binarizer' should tree features be refreshed.
    dump(tcl, tree_gate)

    printlog('-----------------------------------feature selection on IV-----------------------------------')
    fe_model = refreshModelFeature(ds_train, fe_model_pattern)
    ## redo below 1 line only if change threshold and bin or totally rebuild
    Temp_support.cut(ds_train, fe_model, threshold=10, bin=10, method='equal-frequency', save_path=ds_cut, encoding='gb18030')
    Temp_support.select_feature_iv(ds_cut, fe_model, -1, iv_upper_thresh, iv_lower_thresh, to_file=iv_detail, encoding='gb18030')
    ds_temp = pd.read_csv(iv_detail, encoding='gb18030', header=0, index_col=0)['iv']
    ds_temp[ds_temp.between(iv_lower_thresh, iv_upper_thresh)].to_csv(fe_iv, header='iv')

    printlog('-----------------------------------feature selection on lasso-----------------------------------')
    classed_fe_model = Preprocess.pattern_to_feature(ds_train, fe_model_pattern, encoding='gb18030')
    lasso = Lasso(alpha=lasso_alpha)
    ds_t = pd.read_csv(ds_train, encoding='gb18030', header=0, index_col=0)
    listed_fe_lasso = []
    listed_selected_fe = []
    for fe_model in classed_fe_model:
        lasso.fit(ds_t.loc[:, fe_model].values, ds_t.iloc[:, -1].values),
        listed_fe_lasso.append(pd.DataFrame(lasso.coef_, index=fe_model, columns=['lasso']))
        selected_index = np.abs(lasso.coef_) > lasso_coef
        listed_selected_fe.append(listed_fe_lasso[-1][selected_index])
    pd.concat(listed_fe_lasso, axis=0).to_csv(lasso_detail, encoding='gb18030')
    pd.concat(listed_selected_fe, axis=0).to_csv(fe_lasso, encoding='gb18030')

    printlog('-----------------------------------feature selection on xgb-----------------------------------')
    classed_fe_model = Preprocess.pattern_to_feature(ds_train, fe_model_pattern, encoding='gb18030')
    xgb = XGBClassifier()
    ds_t = pd.read_csv(ds_train, encoding='gb18030', header=0, index_col=0)
    listed_fe_xgb = []
    listed_selected_fe = []
    for fe_model in classed_fe_model:
        xgb.fit(ds_t.loc[:, fe_model].values, ds_t.iloc[:, -1].values),
        listed_fe_xgb.append(pd.DataFrame(xgb.feature_importances_, index=fe_model, columns=['lasso']))
        selected_index = xgb.feature_importances_.argsort()[-30:]
        listed_selected_fe.append(listed_fe_xgb[-1].iloc[selected_index, :])
    pd.concat(listed_fe_xgb, axis=0).to_csv(xgb_detail, encoding='gb18030')
    pd.concat(listed_selected_fe, axis=0).to_csv(fe_xgb, encoding='gb18030')

    printlog('-----------------------------------features-----------------------------------')
    hitrate_features  = Log.iterread(fe_gate_hit)
    tree_features     = Log.iterread(fe_gate_tree)
    selected_features = [
      'als_m12_id_nbank_orgnum', 'als_m3_id_cooff_allnum', 
      'ir_id_x_cell_cnt', 'als_m6_id_rel_allnum', 
      'als_fst_id_nbank_inteday', 'cons_tot_m12_visits', 
      'pd_gender_age']

    printlog('-----------------------------------train-----------------------------------')
    printlog('-----------------------------------train on xgb-----------------------------------')
    def objective(y_true, y_pred):
        multiplier = pd.Series(y_true).mask(y_true == 1, xgb_FN_grad_mul).mask(y_true == 0, xgb_FP_grad_mul)
        grad = multiplier * (y_pred - y_true)
        hess = multiplier * np.ones(y_pred.shape)
        return grad, hess
    xgb = XGBClassifier(objective=objective)
    train_dataset = pd.read_csv(ds_train, encoding='gb18030', header=0, index_col=0)
    valid_dataset = pd.read_csv(ds_valid, encoding='gb18030', header=0, index_col=0)
    xgb_params          = {'max_depth': [3, 4, 5], 'n_estimators': range(10, 301, 10)}
    xgb_scorer          = ['neg_mean_squared_error', 'roc_auc']
    xgb_optim_params    = Assess.gridCVSelection(xgb, 'xgb', 'misc', 
        train_dataset.loc[:, selected_features], train_dataset.iloc[:,-1], 
        valid_dataset.loc[:, selected_features], valid_dataset.iloc[:,-1], 
        xgb_params, xgb_scorer, refit_scorer='roc_auc')
    xgb.fit(train_dataset.loc[:, selected_features], train_dataset.iloc[:, -1])
    cutoff = optimalCufoff(xgb, valid_dataset.loc[:, selected_features], valid_dataset.iloc[:, -1].to_numpy())
    dump(xgb, model_xgb)
    xgb.set_params(**xgb_optim_params)
    xgb.fit(train_dataset.loc[:, selected_features], train_dataset.iloc[:, -1])
    prediction = xgb.predict_proba(valid_dataset.loc[:, selected_features])
    cutoff_optim = optimalCufoff(xgb, valid_dataset.loc[:, selected_features], valid_dataset.iloc[:, -1].to_numpy())
    dump(xgb, model_xgb_optim)

    printlog('-----------------------------------test-----------------------------------')
    test_dataset = pd.read_csv(ds_test, encoding='gb18030', header=0, index_col=0)
    printlog('-----------------------------------test on gate and tree-----------------------------------')
    pred_hit     = (test_dataset[hitrate_features] != -1).any(axis=1).astype(int)
    pred_tree    = pd.Series(load(tree_gate).predict(test_dataset[tree_features]), index=test_dataset.index)
    printlog('gate test: {} labelled 1 by hit positive rate.'.format(pred_hit.sum()))
    printlog('gate test: {} labelled 1 by tree classifier.'.format(pred_tree.sum()))
    printlog('-----------------------------------test on xgb-----------------------------------')
    prediction          = load(model_xgb).predict_proba(test_dataset.loc[:, selected_features])
    prediction_optim    = load(model_xgb_optim).predict_proba(test_dataset.loc[:, selected_features])
    ## apply gate prediction to xgb prediction
    prediction[:, 1] += pred_hit + pred_tree
    prediction[prediction[:, 1] > 1, 1] = 1
    prediction_optim[:, 1] += pred_hit + pred_tree
    prediction_optim[prediction_optim[:, 1] > 1, 1] = 1
    ## apply cutoff formula
    prediction[prediction[:, 1] <= cutoff, 1] = 0
    prediction[prediction[:, 1] == 0, 0] = 1
    prediction[prediction[:, 1] == 1, 0] = 0
    prediction_optim[prediction_optim[:, 1] <= cutoff_optim, 1] = 0
    prediction_optim[prediction_optim[:, 1] == 0, 0] = 1
    prediction_optim[prediction_optim[:, 1] == 1, 0] = 0
    ## assess model
    Assess.modelAssess(valid_dataset.iloc[:, -1].to_numpy(), prediction,       'misc', 'XGB')
    Assess.modelAssess(valid_dataset.iloc[:, -1].to_numpy(), prediction_optim, 'misc', 'XGB_optim')

    printlog('-----------------------------------finished-----------------------------------')


def varyDataset(ds, save_path):
    classed_feature_preffix = [['^als_d7_id_', '^als_d15_id_', '^als_m1_id_', '^als_m3_id_', '^als_m6_id_', '^als_m12_id_', '^als_fst_id_', '^als_lst_id_'], ['^als_d7_cell_', '^als_d15_cell_', '^als_m1_cell_', '^als_m3_cell_', '^als_m6_cell_', '^als_m12_cell_', '^als_fst_cell_', '^als_lst_cell_']]
    printlog('class 5 - value padding: larger/smaller')
    ds_t = pd.read_csv(ds, encoding='gb18030', header=0, index_col=0)
    for i, (id_fc, cell_fc) in enumerate(zip(Preprocess.pattern_to_feature(ds_t, 
        classed_feature_preffix[0], encoding='gb18030'), Preprocess.pattern_to_feature(ds_t, 
        classed_feature_preffix[1], encoding='gb18030'))):
        for id_f, cell_f in zip(id_fc, cell_fc):
            ds_t.insert(loc=ds_t.columns.get_loc(id_f), column=id_f.replace('id', 'large'), value=ds_t[[id_f, cell_f]].apply(np.max, axis=1))
            ds_t.insert(loc=ds_t.columns.get_loc(id_f), column=id_f.replace('id', 'small'), value=ds_t[[id_f, cell_f]].apply(np.min, axis=1))
        printlog('class 5 - value padding finished {} and {}'.format(classed_feature_preffix[0][i], classed_feature_preffix[1][i]))
    ds_t.to_csv(save_path, encoding='gb18030')


def refreshModelFeature(ds, listed_feature_pattern):
    fe_temp         = Preprocess.pattern_to_feature(ds, listed_feature_pattern, encoding='gb18030')
    fe_model        = []
    for fe_class in fe_temp:
        fe_model.extend(fe_class)
    return fe_model


def generateExperienceFeature(ds):
    printlog('-----------------------------------generate experience feature-----------------------------------')
    ds_temp = pd.read_csv(ds, encoding='gb18030', header=0, index_col=0)
    series_t = pd.Series(ds_temp['cons_tot_m12_visits'], ds_temp.index)
    series_t[series_t.between(-99.001, -0.001)]    = -99
    series_t[series_t.between(-0.001, 500.001)]    = 500
    series_t[series_t.between(500.001, 1000.001)]  = 1000
    series_t[series_t.between(1000.001, 1500.001)] = 1500
    series_t[series_t.between(1500.001, 900000)]   = 9000
    ds_temp.loc[:, 'cons_tot_m12_visits'] = series_t
    
    series_t = pd.Series(data=-1, index=ds_temp.index)
    series_t[(ds_temp['pd_id_gender'] == 0) & (ds_temp['pd_id_apply_age'].between(-99.001, 30.001))] = 0
    series_t[(ds_temp['pd_id_gender'] == 0) & (ds_temp['pd_id_apply_age'].between(30.001, 60.001))]  = 1
    series_t[(ds_temp['pd_id_gender'] == 0) & (ds_temp['pd_id_apply_age'].between(60.001, 999.001))] = 2
    series_t[(ds_temp['pd_id_gender'] == 1) & (ds_temp['pd_id_apply_age'].between(-0.001, 24.001))]  = 3
    series_t[(ds_temp['pd_id_gender'] == 1) & (ds_temp['pd_id_apply_age'].between(24.001, 35.001))]  = 4
    series_t[(ds_temp['pd_id_gender'] == 1) & (ds_temp['pd_id_apply_age'].between(35.001, 45.001))]  = 5
    series_t[(ds_temp['pd_id_gender'] == 1) & (ds_temp['pd_id_apply_age'].between(45.001, 999.001))] = 2
    if 'pd_gender_age' not in ds_temp.columns:
        ds_temp.insert(ds_temp.columns.size - 1, 'pd_gender_age', series_t)
    else:
        ds_temp.loc[:, 'pd_gender_age'] = series_t
    ds_temp.to_csv(ds, encoding='gb18030')
    

def optimalCufoff(estimator, features, labels):
    assessment = []
    for cutoff in [i / 10 for i in range(1, 9)]:
        pred = estimator.predict_proba(features)[:, 1]
        pred[pred >  cutoff] = 1
        pred[pred <= cutoff] = 0
        true_pos  = ((pred == 1) & (labels == 1)).sum()
        false_pos = ((pred == 1) & (labels == 0)).sum()
        assessment.append(true_pos * 0.1 - false_pos * 0.6 - 200 / (true_pos + false_pos))
    printlog('optimalCutoff: {}'.format((np.array(assessment).argmax() + 1) / 10))
    return (np.array(assessment).argmax() + 1) / 10


if __name__ == '__main__':
    try:
        run()
        potplayer.run('jinitaimei.m4a')
    except Exception as e:
        logging.error(traceback.format_exc())
        potplayer.run('oligei.m4a')