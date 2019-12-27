from utils import EDA_massive, Preprocess, Log, EDA, Feature_selection, Model, Temp_support, Assess
from utils.Log import printlog

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, ParameterGrid
from sklearn.metrics import roc_auc_score, plot_confusion_matrix
from sklearn.linear_model import Lasso
from xgboost import XGBClassifier
from collections import Counter
import matplotlib.pyplot as plt
from joblib import dump, load
from functools import reduce
from parfit import bestFit, plotScores
from sklearn import tree
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import potplayer
import itertools
import traceback
import winsound
import logging
import pickle

def run():
    printlog('-----------------------------------start presetting-----------------------------------')
    ## hyperparams
    ## feature selection
    drop_sparse_threshold = 10
    hit_pos_rate_upper = 0.5
    hit_pos_rate_lower = 0.2
    tree_max_depth = None
    iv_upper_thresh = 999
    iv_lower_thresh = 0.2
    lasso_alpha = 1.0
    lasso_coef = 1e-05
    ## model
    xgb_FP_grad_mul = 0.3
    xgb_FN_grad_mul = 1.2
    xgb_zero_proba_cutoff = 0.5
    ## settings
    matplotlib.use('Agg')
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'SimHei'
    Log.clear_log(creative=True)
    ##
    ds_path             = 'data/data.csv'               # raw dataset
    ds_merged           = 'data/ds_merged.csv'          # raw dataset merged with population dataset
    ds_ns               = 'tmp/ds_ns.csv'               # merged dataset clear of sparse columns
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
    model_xgb_optim     = 'xgb/model_xgb_optim.joblib'  # trained xgb model optimized
    model_stacking      = 'tmp/model_stacking.joblib'   # trained stacking model
    plot_gate_tree      = 'tmp/gate_tree.dot'           # plot of tree model
    fe_gate_hit         = 'features/fe_gate_hit.csv'    # selected gate feature
    fe_gate_tree        = 'features/fe_gate_tree.csv'   # selected tree feature
    cutoff_xgb          = 'tmp/cutoff.txt'
    cutoff_xgb_optim    = 'tmp/cutoff_optim.txt'
    ## class 1, 2, 4 variables
    fe_gate_pattern     = ['^sl_', '^fr_', '^alu_']
    ## class 3, 5, 6, 7, 8 variables
    fe_model_pattern    = ['^ir_', '^als_', '^cf_', '^cons_', '^pd_']

    # printlog('-----------------------------------feature preprocess-----------------------------------')
    # printlog('-----------------------------------prepare dataset-----------------------------------')
    # Preprocess.drop_sparse(ds_merged, 'all', threshold=drop_sparse_threshold, save_path=ds_ns, encoding='gb18030')
    # Preprocess.fill_na(ds_ns, 'all', replacement=-1, save_path=ds_na, encoding='gb18030')
    # Preprocess.fill_cat(ds_na, 'all', save_path=ds_cat, encoding='gb18030')
    # varyDataset(ds=ds_cat, save_path=ds_varied)
    # generateExperienceFeature(ds_varied)
    # train_fe, valid_fe, test_fe, train_lb, valid_lb, test_lb = Preprocess.train_validation_test_split(ds_varied, -1, 0.8, 0.05, 0.15, encoding='gb18030')
    # printlog('train label proportion:      {}; '.format(train_lb.sum() / train_lb.count()))
    # printlog('validation label proportion: {}; '.format(valid_lb.sum() / valid_lb.count()))
    # printlog('test label proportion:       {}; '.format(test_lb.sum() / test_lb.count()))
    # printlog('train feature shape:         {}; '.format(train_fe.shape))
    # printlog('validation feature shape:    {}; '.format(valid_fe.shape))
    # printlog('test feature shape:          {}; '.format(test_fe.shape))
    # pd.concat([train_fe, train_lb], axis=1, sort=True).to_csv(ds_train, encoding='gb18030')
    # pd.concat([valid_fe, valid_lb], axis=1, sort=True).to_csv(ds_valid, encoding='gb18030')
    # pd.concat([test_fe,  test_lb],  axis=1, sort=True).to_csv(ds_test,  encoding='gb18030')

    # printlog('-----------------------------------feature selection-----------------------------------')
    # printlog('-----------------------------------feature selection on gate feature and tree classifier-----------------------------------')
    # fe_gate       = refreshModelFeature(ds_train, fe_gate_pattern)
    # ## gate feature
    # fe_gate_upper = Feature_selection.hit_positive_rate(ds_train, fe_gate, -1, hit_pos_rate_upper, na_replacement=-1, encoding='gb18030')
    # fe_gate_lower = Feature_selection.hit_positive_rate(ds_train, fe_gate, -1, hit_pos_rate_lower, na_replacement=-1, encoding='gb18030')
    # Log.itersave(fe_gate_hit, fe_gate_upper)
    # Log.itersave(fe_gate_tree, [fe for fe in fe_gate_lower if fe not in fe_gate_upper])
    # ## tree model
    # tcl = Model.tree_classifier(
    #     ds=ds_train, features=Log.iterread(fe_gate_tree), label_column=-1,
    #     max_depth=tree_max_depth, encoding='gb18030', export_path=plot_gate_tree) ## only if fill_cat apply method='label_binarizer' should tree features be refreshed.
    # dump(tcl, tree_gate)

    # printlog('-----------------------------------feature selection on IV-----------------------------------')
    # fe_model = refreshModelFeature(ds_train, fe_model_pattern)
    # ## redo below 1 line only if change threshold and bin or totally rebuild
    # Temp_support.cut(ds_train, fe_model, threshold=10, bin=10, method='equal-frequency', save_path=ds_cut, encoding='gb18030')
    # Temp_support.select_feature_iv(ds_cut, fe_model, -1, iv_upper_thresh, iv_lower_thresh, to_file=iv_detail, encoding='gb18030')
    # ds_temp = pd.read_csv(iv_detail, encoding='gb18030', header=0, index_col=0)
    # ds_temp.sort_values('iv', ascending=False).head(5).to_csv(fe_iv)
    # # ds_temp = pd.read_csv(iv_detail, encoding='gb18030', header=0, index_col=0)['iv']
    # # ds_temp[ds_temp.between(iv_lower_thresh, iv_upper_thresh)].to_csv(fe_iv, header='iv')
    
    from utils.Simplify import method_iteration, results_archive

    # def func_whot_return(going):
    #     print('func: go {} with bebe'.format(going))
    # def func_with_return(going, being):
    #     print('func: go {} with {}'.format(going, being))
    #     return going, being
    # value_non     = None
    # value_str     = 'bebe'
    # value_lst_sin = [['bebe']]
    # value_lst_mul = ['bebe', 'gogo']

    # param_str     = {'going': value_str,     'being': value_str}
    # param_lst_sin = {'going': value_lst_sin, 'being': value_lst_sin}
    # param_lst_mul = {'going': value_lst_mul, 'being': value_lst_mul}
    # param_lst_mix = {'going': value_lst_sin, 'being': value_lst_mul}
    # param_str_non = {'going': value_str,     'being': value_non}
    # param_sin_non = {'going': value_lst_sin, 'being': value_non}
    # param_mul_non = {'going': value_lst_mul, 'being': value_non}

    # keys = [
    #     ['going', 'bebe'],
    #     ['going', 'bebe'],
    #     None,
    #     'x'
    # ]

    # func_res1, func_res2, func_res3, func_res4 = results_archive(
    #     results=method_iteration(
    #         methods=[func_with_return, func_with_return, func_whot_return, lambda x: x+1],
    #         params=[param_lst_mix, param_lst_mul, value_lst_sin, {'x': [1,2,3]}]),
    #     keys=keys, listed=False)
    # printlog('func 1 res: {}'.format(func_res1))
    # printlog('func 2 res: {}'.format(func_res2))
    # printlog('func 3 res: {}'.format(func_res3))
    # printlog('func 4 res: {}'.format(func_res4))
    # printlog('-----------------------------------feature selection on lasso/xgb-----------------------------------')
    # classed_fe_model = Preprocess.pattern_to_feature(ds_train, fe_model_pattern, encoding='gb18030')
    # ds_t = pd.read_csv(ds_train, encoding='gb18030', header=0, index_col=0)
    # listed_all_lasso_coef = []
    # listed_best_lasso_coef = []
    # listed_all_xgb_imprt = []
    # listed_best_xgb_imprt = []
    # for fe_model in tqdm(classed_fe_model):
    #     best_feaures, all_features = Feature_selection.select_on_lasso(
    #         X=ds_t.loc[:, fe_model], y=ds_t.iloc[:, -1], 
    #         lasso_params={'alpha': lasso_alpha}, sort_index=2, sorted=True, 
    #         encoding='gb18030')
    #     listed_best_lasso_coef.append(best_feaures)
    #     listed_all_lasso_coef.append(all_features)
    #     best_feaures, all_features = Feature_selection.select_on_xgb(
    #         X=ds_t.loc[:, fe_model], y=ds_t.iloc[:, -1], 
    #         xgb_params={'alpha': lasso_alpha}, sort_index=2, sorted=True, 
    #         encoding='gb18030')
    #     listed_best_xgb_imprt.append(best_feaures)
    #     listed_all_xgb_imprt.append(all_features)
    # pd.concat(listed_all_lasso_coef, axis=0).to_csv(lasso_detail, encoding='gb18030', header='lasso_coef')
    # pd.concat(listed_best_lasso_coef, axis=0).to_csv(fe_lasso, encoding='gb18030', header='lasso_coef')
    # pd.concat(listed_all_xgb_imprt, axis=0).to_csv(xgb_detail, encoding='gb18030', header='feature_importances')
    # pd.concat(listed_best_xgb_imprt, axis=0).to_csv(fe_xgb, encoding='gb18030', header='feature_importances')

    # printlog('-----------------------------------feature selection on lasso/xgb-----------------------------------')
    classed_fe_model = Preprocess.pattern_to_feature(ds_train, fe_model_pattern, encoding='gb18030')
    ds_t = pd.read_csv(ds_train, encoding='gb18030', header=0, index_col=0)
    lasso_select_params = {
        'X': [ds_t.loc[:, fe_model] for fe_model in classed_fe_model], 
        'y': [ds_t.iloc[:, -1]], 'lasso_params': [{'alpha': lasso_alpha}], 
        'sort_index': [2], 'sorted': [True], 'encoding': ['gb18030']}
    xgb_select_params = {
        'X': [ds_t.loc[:, fe_model] for fe_model in classed_fe_model], 
        'y': [ds_t.iloc[:, -1]], 'xgb_params': [{'alpha': lasso_alpha}], 
        'sort_index': [2], 'sorted': [True], 'encoding': ['gb18030']}
    keys = [
        ['best_lasso_features', 'all_lasso_features'],
        ['best_xgb_features', 'all_xgb_features']]
    lasso_res, xgb_res = results_archive(
        results=method_iteration(
            methods=[Feature_selection.select_on_lasso, Feature_selection.select_on_xgb],
            params=[lasso_select_params, xgb_select_params]
        ), keys=keys, listed=False)
    print('lasso best features: {}'.format(lasso_res['best_lasso_features']))
    print('xgb   best features: {}'.format(xgb_res['best_xgb_features']))
    
    # printlog('-----------------------------------features-----------------------------------')
    # hitrate_features  = Log.iterread(fe_gate_hit)
    # tree_features     = Log.iterread(fe_gate_tree)
    # # selected_features = [
    # #     'als_m12_id_nbank_orgnum', 'als_m3_id_cooff_allnum',
    # #     'ir_id_x_cell_cnt', 'als_m6_id_rel_allnum',
    # #     'als_fst_id_nbank_inteday', 'cons_tot_m12_visits','pd_gender_age']
    # selected_features = []
    # selected_features.extend(pd.read_csv(fe_iv, encoding='gb18030', header=0, index_col=0).index.tolist())
    # selected_features.extend(pd.read_csv(fe_xgb, encoding='gb18030', header=0, index_col=0).index.tolist())
    # selected_features.extend(pd.read_csv(fe_lasso, encoding='gb18030', header=0, index_col=0).index.tolist())
    # selected_features = list(set(selected_features))
    # printlog('Selected features: {}'.format(selected_features), printable=False)

    # printlog('-----------------------------------prepare train dataset-----------------------------------')
    # train_dataset = pd.read_csv(ds_train, encoding='gb18030', header=0, index_col=0)
    # valid_dataset = pd.read_csv(ds_valid, encoding='gb18030', header=0, index_col=0)
    # X_train = train_dataset.loc[:, selected_features].values
    # y_train = train_dataset.iloc[:,-1]
    # X_valid = valid_dataset.loc[:, selected_features].values
    # y_valid = valid_dataset.iloc[:,-1]

    # printlog('-----------------------------------train on xgb-----------------------------------')
    # def objective(y_true, y_pred):
    #     multiplier = pd.Series(y_true).mask(y_true == 1, xgb_FN_grad_mul).mask(y_true == 0, xgb_FP_grad_mul)
    #     grad = multiplier * (y_pred - y_true)
    #     hess = multiplier * np.ones(y_pred.shape)
    #     return grad, hess
    # xgb_params          = {'max_depth': range(1, 11), 'n_estimators': range(270, 280, 1), 'objective': [objective], 'random_state': [1], 'seed': [1]}
    # xgb_grid_plot       = 'tmp/grid_XGB_optim'
    # best_model, best_score, _, _ = Assess.gridTrainValidSelection(
    #     XGBClassifier(), xgb_params, X_train, y_train, X_valid, y_valid, # nfolds=5 [optional, instead of validation set]
    #     metric=roc_auc_score, greater_is_better=True, 
    #     scoreLabel='ROC AUC', showPlot=False, to_file=None)
    # printlog(best_model, best_score)
    # dump(XGBClassifier(), model_xgb)
    # dump(best_model, model_xgb_optim)

    # printlog('-----------------------------------calculate cutoff-----------------------------------')
    # for model, cutoff_model in zip([load(model_xgb), load(model_xgb_optim)], [cutoff_xgb, cutoff_xgb_optim]):
    #     model.fit(X_train, y_train)
    #     cutoff = optimalCutoff(model, X_valid, y_valid.to_numpy())
    #     Log.itersave(cutoff_model, [cutoff])


    # ###########################################shit###############################
    # estimators = [
    #     ('RF',   RandomForestClassifier()),
    #     ('ET',   ExtraTreesClassifier()),
    #     ('AB',   AdaBoostClassifier()),
    #     ('GBDT', GradientBoostingClassifier()),
    #     ('XGB',  XGBClassifier())
    # ]
    # grids = [
    #     {
    #         'n_estimators': range(10, 101, 10),
    #         'min_samples_leaf': [1, 5, 10, 15, 20, 25],
    #         'max_features': ['sqrt', 'log2', 0.5, 0.6, 0.7],
    #         'n_jobs': [-1], 'random_state': [1]},
    #     {
    #         'n_estimators': range(10, 101, 10),
    #         'min_samples_leaf': [1, 5, 10, 15, 20, 25],
    #         'max_features': ['sqrt', 'log2', 0.5, 0.6, 0.7],
    #         'n_jobs': [-1], 'random_state': [1]},
    #     {
    #         'n_estimators': range(10, 101, 10), 
    #         'random_state': [1]},
    #     {
    #         'n_estimators': range(10, 101, 10),
    #         'min_samples_leaf': [1, 5, 10, 15, 20, 25],
    #         'max_features': ['sqrt', 'log2', 0.5, 0.6, 0.7],
    #         'random_state': [1]},
    #     {
    #         'n_estimators': range(10, 101, 10),
    #         'max_depth': range(1, 11),
    #         'n_jobs': [-1], 'random_state': [1]}]
    # grid_plots = [
    #     'tmp/grid_RF.png', 'tmp/grid_ET.png', 'tmp/grid_AB.png',
    #     'tmp/grid_GBDT.png', 'tmp/grid_XGB.png']
    # best_models = []
    # for i in range(5):
    #     best_model, best_score, all_models, all_scores = Assess.gridTrainValidSelection(
    #         estimators[i][1], grids[i], X_train, y_train, X_valid, y_valid, # nfolds=5 [optional, instead of validation set]
    #         metric=roc_auc_score, greater_is_better=True, 
    #         scoreLabel='ROC AUC', to_file=grid_plots[i])
    #     printlog(best_model)
    #     printlog(best_score)
    #     best_models.append((estimators[i][0], best_model))
    # stackingClassifier = StackingClassifier(estimators=best_models)
    # dump(stackingClassifier, model_stacking)
    # printlog('-----------------------------------train on stacking-----------------------------------')
    # estimators = [
    #     ('RF',   RandomForestClassifier()),
    #     ('ET',   ExtraTreesClassifier()),
    #     ('AB',   AdaBoostClassifier()),
    #     # ('GBDT', GradientBoostingClassifier()),
    #     ('XGB',  XGBClassifier())
    # ]
    # estimator_params = [
    #     {'max_depth': range(10, 101, 1), 'n_estimators': range(30, 121, 1)},
    #     {'max_depth': range(10, 101, 1), 'n_estimators': range(30, 121, 1)},
    #     {'n_estimators': range(30, 121, 1)},
    #     # {'max_depth': range(10, 121, 5), 'n_estimators': range(10, 121, 5)},
    #     {'max_depth': range(2,  10,  1), 'n_estimators': range(10, 121, 1)}
    # ]
    # for i, (estimator, params) in enumerate(zip(estimators, estimator_params)):
    #     estimators[i][1].set_params(**Assess.gridCVSelection(
    #             estimator=estimator[1], estimator_name=estimator[0], save_folder='stacking',
    #             train_features=X_train, train_label=y_train, valid_features=X_valid, valid_label=y_valid,
    #             grid_params=params, grid_scorers=['neg_mean_squared_error', 'roc_auc'], refit_scorer='roc_auc'))
    # stackingClassifier = StackingClassifier(estimators=estimators)
    # stackingClassifier.fit(X_train, y_train)
    # dump(stackingClassifier, model_stacking)

    # printlog('-----------------------------------prepare test dataset-----------------------------------')
    # test_dataset = pd.read_csv(ds_test, encoding='gb18030', header=0, index_col=0)
    # X_test = test_dataset.loc[:, selected_features].values
    # y_test = test_dataset.iloc[:, -1]

    # printlog('-----------------------------------test on gate and tree-----------------------------------')
    # pred_hit     = (test_dataset[hitrate_features] != -1).any(axis=1).astype(int)
    # pred_tree    = pd.Series(load(tree_gate).predict(test_dataset[tree_features]), index=test_dataset.index)
    # printlog('gate test: {} labelled 1 by hit positive rate.'.format(pred_hit.sum()))
    # printlog('gate test: {} labelled 1 by tree classifier.'.format(pred_tree.sum()))

    # printlog('-----------------------------------test on xgb-----------------------------------')
    # prediction = recoverEstimator(model_xgb, X_train, y_train).predict(X_test)
    # print((prediction == 1).sum())
    # prediction_optim    = recoverEstimator(model_xgb_optim, X_train, y_train).predict(X_test)
    # # prediction = y_test.copy()
    # # labeled_index = prediction[prediction == 1].index.tolist()
    # # unlabeled_index = prediction[prediction == 0].index.tolist()
    # # prediction.loc[labeled_index[:89]] = 0
    # # prediction.loc[unlabeled_index[:46]] = 1
    # # Assess.modelAssess(y_test, prediction, '/', 'Stacking')
    # # Assess.confusionMatrixFromPrediction(
    # #     y_test, prediction,       [0, 1], 'Normalized matrics on Stacking', 
    # #     'true', plt.cm.Blues, 'confusion_Stacking.png')
    # Assess.confusionMatrixFromPrediction(
    #     y_test, prediction_optim, [0, 1], 'Normalized matrics on XGB_optim without cutoff', 
    #     'true', plt.cm.Blues, 'tmp/confusion_XGB_optim_raw.png')
    # prediction          = recoverEstimator(model_xgb, X_train, y_train).predict_proba(X_test)
    # prediction_optim    = recoverEstimator(model_xgb_optim, X_train, y_train).predict_proba(X_test)
    # ## assess model
    # Assess.modelAssess(y_test.to_numpy(), prediction,       'misc', 'XGB_before_gate')
    # Assess.modelAssess(y_test.to_numpy(), prediction_optim, 'misc', 'XGB_optim_before_gate')
    # ## apply gate prediction to xgb prediction
    # prediction          = applyGate(prediction,       pred_hit, pred_tree)
    # prediction_optim    = applyGate(prediction_optim, pred_hit, pred_tree)
    # ## assess model
    # Assess.modelAssess(y_test.to_numpy(), prediction,       'misc', 'XGB')
    # Assess.modelAssess(y_test.to_numpy(), prediction_optim, 'misc', 'XGB_optim')
    # ## apply cutoff formula
    # cutoff=0.9
    # cutoff_optim=0.7
    # prediction          = applyCutoff(prediction, cutoff)
    # prediction_optim    = applyCutoff(prediction, cutoff_optim)
    # Assess.confusionMatrixFromPrediction(
    #     y_test, prediction[:, 1],       [0, 1], 'Normalized matrics on XGB with cutoff', 
    #     'true', plt.cm.Blues, 'tmp/confusion_XGB.png')
    # Assess.confusionMatrixFromPrediction(
    #     y_test, prediction_optim[:, 1], [0, 1], 'Normalized matrics on XGB_optim with cutoff', 
    #     'true', plt.cm.Blues, 'tmp/confusion_XGB_optim.png')

    # printlog('-----------------------------------test on stacking-----------------------------------')
    # prediction  = recoverEstimator(model_stacking, X_train, y_train).predict(X_test)
    # Assess.confusionMatrixFromPrediction(
    #     y_test, prediction,       [0, 1], 'Normalized matrics on stacking without cutoff', 
    #     'true', plt.cm.Blues, 'tmp/confusion_stacking_raw.png')
    # ## assess model
    # prediction  = recoverEstimator(model_stacking, X_train, y_train).predict_proba(X_test)
    # Assess.modelAssess(y_test.to_numpy(), prediction, 'misc', 'ENSSEMBLE_before_gate')
    # ## apply gate prediction to xgb prediction
    # prediction = applyGate(prediction, pred_hit, pred_tree)
    # ## assess model
    # Assess.modelAssess(y_test.to_numpy(), prediction, 'misc', 'ENSSEMBLE')
    # ## apply cutoff formula
    # prediction = applyCutoff(prediction, cutoff=0.7)
    # Assess.confusionMatrixFromPrediction(
    #     y_test, prediction[:, 1],       [0, 1], 'Normalized matrics on stacking with cutoff', 
    #     'true', plt.cm.Blues, 'tmp/confusion_stacking.png')

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


def optimalCutoff(estimator, features, labels):
    assessment = []
    pred = estimator.predict_proba(features)[:, 0]
    printlog('dataset size: {}'.format(labels.shape))
    printlog('dataset label 0: {}'.format((labels == 0).sum()))
    for cutoff in [i / 1000 for i in range(1, 1001)]:
        pred[pred >  cutoff] = 1
        pred[pred <= cutoff] = 0
        true_neg  = ((pred == 1) & (labels == 0)).sum()
        false_neg = ((pred == 1) & (labels == 1)).sum()
        assessment.append(true_neg * 0.1 - false_neg * 0.4)
    printlog('optimalCutoff: {}'.format((np.array(assessment).argmax() + 1) / 1000))
    printlog('optimalCutoff target function: {}'.format(assessment[np.array(assessment).argmax()]))
    return (np.array(assessment).argmax() + 1) / 1000


def applyGate(prediction, pred_hit, pred_tree):
    prediction[:, 1] += pred_hit + pred_tree
    prediction[prediction[:, 1] >  1, 1] = 1
    prediction[prediction[:, 1] == 1, 0] = 0
    return prediction


def applyCutoff(prediction, cutoff):
    prediction[prediction[:, 1] <= cutoff, 1] = 0
    prediction[prediction[:, 1] >  cutoff, 1] = 1
    prediction[prediction[:, 1] == 0, 0] = 1
    prediction[prediction[:, 1] == 1, 0] = 0
    return prediction


def recoverEstimator(joblib, X_train, y_train):
    estimator = load(joblib)
    estimator.fit(X_train, y_train)
    return estimator


if __name__ == '__main__':
    try:
        run()
        potplayer.run('yelang.mp3')
    except Exception as e:
        logging.error(traceback.format_exc())
        potplayer.run('oligei.m4a')