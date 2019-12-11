from utils import EDA_massive, Preprocess, Log, EDA, Feature_selection, Model, Temp_support, Assess
from utils.Log import printlog

from sklearn.model_selection import train_test_split, cross_val_score
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
import winsound

def run():
    ## hyperparams
    ## feature selection
    hit_pos_rate_upper = 0.5
    hit_pos_rate_lower = 0.2
    tree_max_depth = 3
    iv_upper_thresh = 0.5
    iv_lower_thresh = 0.2
    lasso_alpha = 1.0
    lasso_coef = 1e-04
    ## model
    xgb_FP_grad_mul = 1.0
    xgb_FN_grad_mul = 1.2
    ## settings
    ds_path = 'data/data.csv'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'SimHei'
    Log.clear_log(creative=True)
    ''' data flow:
        ds_c[n]                 raw dataset
        ds_c[n]_na              after na data are filled
        ds_c[n]_cut[1/2]        after data are cut by method 1/2
        iv_c[n]_cut[1/2]        dataset of features as rows and IVs as columns
        fe_c[n]_cut[1/2]_iv     dataset of features as rows and IVs as columns
        fe_c[n]_pattern         prefix pattern for class features
        fe_c[n]                 list of class features strings
    '''
    ## class 1, 2, 4 variables
    ds_gate         = 'tmp/ds_gate.csv'
    ds_gate_cat     = 'tmp/ds_gate_cat.csv'
    ds_gate_na      = 'tmp/ds_gate_na.csv'
    fe_gate_hit     = 'features/fe_gate_hit.csv'
    fe_gate_tree    = 'features/fe_gate_tree.csv'
    hit_threshold   = 0.5
    tree_threshold  = 0.3
    fe_gate_pattern = ['^sl_', '^fr_', '^alu_']
    fe_gate_t       = Preprocess.pattern_to_feature(ds_path, fe_gate_pattern, encoding='gb18030')
    fe_gate         = []
    for fe_class in fe_gate_t:
        fe_gate.extend(fe_class)
    plot_gate_tree  = 'tmp/gate_tree.dot'
    tree_gate       = 'tmp/tree_gate.joblib'
    ## class 3 variables
    ds_c3           = 'tmp/ds_c3.csv'
    ds_c3_na        = 'tmp/ds_c3_na.csv'
    ds_c3_cut1      = 'tmp/ds_c3_cut1.csv'
    ds_c3_cut2      = 'tmp/ds_c3_cut2.csv'
    iv_c3_cut1      = 'iv/iv_c3_cut1.csv'
    iv_c3_cut2      = 'iv/iv_c3_cut2.csv'
    fe_c3_cut1_iv   = 'features/fe_c3_cut1_iv.csv'
    fe_c3_cut2_iv   = 'features/fe_c3_cut2_iv.csv'
    fe_c3_pattern   = '^ir_'
    fe_c3           = Preprocess.pattern_to_feature(ds_path, fe_c3_pattern, encoding='gb18030')[0]
    fe_c3_lasso     = 'features/fe_c3_lasso.csv'
    fe_c3_xgb       = 'features/fe_c3_xgb.csv'
    lasso_c3        = 'lasso/lasso_c3.csv'
    xgb_c3          = 'xgb/xgb_c3.csv'
    ## class 5 variables
    ds_c5           = 'tmp/ds_c5.csv'
    ds_c5_varied    = 'tmp/ds_c5_varied.csv'
    ds_c5_na        = 'tmp/ds_c5_na.csv'
    ds_c5_cut1      = 'tmp/ds_c5_cut1.csv'
    ds_c5_cut2      = 'tmp/ds_c5_cut2.csv'
    iv_c5_cut1      = 'iv/iv_c5_cut1.csv'
    iv_c5_cut2      = 'iv/iv_c5_cut2.csv'
    fe_c5_cut1_iv   = 'features/fe_c5_cut1_iv.csv'
    fe_c5_cut2_iv   = 'features/fe_c5_cut2_iv.csv'
    als_preffix     = [['^als_d7_id_', '^als_d15_id_', '^als_m1_id_', '^als_m3_id_', '^als_m6_id_', '^als_m12_id_', '^als_fst_id_', '^als_lst_id_'], ['^als_d7_cell_', '^als_d15_cell_', '^als_m1_cell_', '^als_m3_cell_', '^als_m6_cell_', '^als_m12_cell_', '^als_fst_cell_', '^als_lst_cell_']]
    fe_c5_pattern   = '^als_'
    fe_c5           = Preprocess.pattern_to_feature(ds_path, fe_c5_pattern, encoding='gb18030')[0]
    fe_c5_lasso     = 'features/fe_c5_lasso.csv'
    fe_c5_xgb       = 'features/fe_c5_xgb.csv'
    lasso_c5        = 'lasso/lasso_c5.csv'
    xgb_c5          = 'xgb/xgb_c5.csv'
    ## class 6 variables
    ds_c6           = 'tmp/ds_c6.csv'
    ds_c6_na        = 'tmp/ds_c6_na.csv'
    ds_c6_cut1      = 'tmp/ds_c6_cut1.csv'
    ds_c6_cut2      = 'tmp/ds_c6_cut2.csv'
    iv_c6_cut1      = 'iv/iv_c6_cut1.csv'
    iv_c6_cut2      = 'iv/iv_c6_cut2.csv'
    fe_c6_cut1_iv   = 'features/fe_c6_cut1_iv.csv'
    fe_c6_cut2_iv   = 'features/fe_c6_cut2_iv.csv'
    fe_c6_pattern   = '^cf_'
    fe_c6           = Preprocess.pattern_to_feature(ds_path, fe_c6_pattern, encoding='gb18030')[0]
    fe_c6_lasso     = 'features/fe_c6_lasso.csv'
    fe_c6_xgb       = 'features/fe_c6_xgb.csv'
    lasso_c6        = 'lasso/lasso_c6.csv'
    xgb_c6          = 'xgb/xgb_c6.csv'
    ## class 7 variables
    ds_c7           = 'tmp/ds_c7.csv'
    ds_c7_cat       = 'tmp/ds_c7_cat.csv'
    ds_c7_na        = 'tmp/ds_c7_na.csv'
    ds_c7_cut1      = 'tmp/ds_c7_cut1.csv'
    ds_c7_cut2      = 'tmp/ds_c7_cut2.csv'
    iv_c7_cut1      = 'iv/iv_c7_cut1.csv'
    iv_c7_cut2      = 'iv/iv_c7_cut2.csv'
    fe_c7_cut1_iv   = 'features/fe_c7_cut1_iv.csv'
    fe_c7_cut2_iv   = 'features/fe_c7_cut2_iv.csv'
    fe_c7_pattern   = '^cons_'
    fe_c7           = Preprocess.pattern_to_feature(ds_path, fe_c7_pattern, encoding='gb18030')[0]
    fe_c7_lasso     = 'features/fe_c7_lasso.csv'
    fe_c7_xgb       = 'features/fe_c7_xgb.csv'
    lasso_c7        = 'lasso/lasso_c7.csv'
    xgb_c7          = 'xgb/xgb_c7.csv'
    ## class 8 variables
    ds_c8           = 'data/pop.csv'
    ds_c8_na        = 'tmp/ds_c8_na.csv'
    ds_c8_cut1      = 'tmp/ds_c8_cut1.csv'
    ds_c8_cut2      = 'tmp/ds_c8_cut2.csv'
    iv_c8_cut1      = 'iv/iv_c8_cut1.csv'
    iv_c8_cut2      = 'iv/iv_c8_cut2.csv'
    fe_c8_cut1_iv   = 'features/fe_c8_cut1_iv.csv'
    fe_c8_cut2_iv   = 'features/fe_c8_cut2_iv.csv'
    fe_c8_pattern   = '^pd_'
    fe_c8           = Preprocess.pattern_to_feature(ds_c8, fe_c8_pattern, encoding='gb18030')[0] 
    fe_c8_lasso     = 'features/fe_c8_lasso.csv'
    fe_c8_xgb       = 'features/fe_c8_xgb.csv'
    lasso_c8        = 'lasso/lasso_c8.csv'
    xgb_c8          = 'xgb/xgb_c8.csv'
    ## experience variables
    ds_exp_na       = 'tmp/ds_exp_na.csv'
    ## classed variables
    classed_ds_na   = [ds_c3_na, ds_c5_na, ds_c6_na, ds_c7_na, ds_c8_na]
    classed_ds_cut1 = [ds_c3_cut1, ds_c5_cut1, ds_c6_cut1, ds_c7_cut1, ds_c8_cut1]
    classed_fe_iv   = [fe_c3_cut1_iv, fe_c5_cut1_iv, fe_c6_cut1_iv, fe_c7_cut1_iv, fe_c8_cut1_iv]
    classed_fe_lasso= [fe_c3_lasso, fe_c5_lasso, fe_c6_lasso, fe_c7_lasso, fe_c8_lasso]
    classed_fe_xgb  = [fe_c3_xgb, fe_c5_xgb, fe_c6_xgb, fe_c7_xgb, fe_c8_xgb]
    classed_lasso   = [lasso_c3, lasso_c5, lasso_c6, lasso_c7, lasso_c8]
    classed_xgb     = [xgb_c3, xgb_c5, xgb_c6, xgb_c7, xgb_c8]
    classed_iv      = [iv_c3_cut1, iv_c5_cut1, iv_c6_cut1, iv_c7_cut1, iv_c8_cut1]


    # printlog('-----------------------------------feature preprocess-----------------------------------')
    # printlog('-----------------------------------class 1, 2, 4 - sl, fr, alu-----------------------------------')
    # ## extract class and label feature
    # ds_t = pd.read_csv(ds_path, encoding='gb18030', header=0, index_col=0)
    # pd.concat([ds_t.loc[:, fe_gate], ds_t.iloc[:, -1]], axis=1, sort=True).to_csv(ds_gate, encoding='gb18030')
    # ## gate feature
    # fe_gate_t = Feature_selection.hit_positive_rate(ds_gate, fe_gate, -1, hit_pos_rate_upper, encoding='gb18030')
    # fe_gate_t1 = Feature_selection.hit_positive_rate(ds_gate, fe_gate, -1, hit_pos_rate_lower, encoding='gb18030')
    # Log.itersave(fe_gate_hit, fe_gate_t)
    # Log.itersave(fe_gate_tree, [fe for fe in fe_gate_t1 if fe not in fe_gate_t])
    # ## tree model
    # fill_na = lambda x, f: Preprocess.fill_na(x, f, replacement=-1, save_path=ds_gate_na, encoding='gb18030')
    # fill_cat = lambda x, f: Preprocess.fill_cat(x, f, save_path=ds_gate_cat, encoding='gb18030')
    # tcl, _, fe_gate_t = Model.tree_classifier(
    #     ds=ds_gate, 
    #     features=Log.iterread(fe_gate_tree), 
    #     label_column=-1, 
    #     max_depth=tree_max_depth, 
    #     fill_na=fill_na, 
    #     fill_cat=fill_cat, 
    #     encoding='gb18030', 
    #     export_path=plot_gate_tree
    # )
    # Log.itersave(fe_gate_tree, fe_gate_t)
    # dump(tcl, tree_gate)
    # printlog('-----------------------------------class 5 - als-----------------------------------')
    # ## extract class and label feature
    # ds_t = pd.read_csv(ds_path, encoding='gb18030', header=0, index_col=0)
    # pd.concat([ds_t.loc[:, fe_c5], ds_t.iloc[:, -1]], axis=1).to_csv(ds_c5, encoding='gb18030')
    # ## start of variation
    # printlog('class 5 - value padding: larger/smaller')
    # ds_t = pd.read_csv(ds_c5, encoding='gb18030', header=0, index_col=0)
    # for i, (id_fc, cell_fc) in enumerate(zip(Preprocess.pattern_to_feature(ds_t, als_preffix[0], encoding='gb18030'), Preprocess.pattern_to_feature(ds_t, als_preffix[1], encoding='gb18030'))):
    #     for id_f, cell_f in zip(id_fc, cell_fc):
    #         ds_t.insert(loc=ds_t.columns.get_loc(id_f), column=id_f.replace('id', 'large'), value=ds_t[[id_f, cell_f]].apply(np.max, axis=1))
    #         ds_t.insert(loc=ds_t.columns.get_loc(id_f), column=id_f.replace('id', 'small'), value=ds_t[[id_f, cell_f]].apply(np.min, axis=1))
    #     printlog('class 5 - value padding finished {} and {}'.format(als_preffix[0][i], als_preffix[1][i]))
    # ds_t.to_csv(ds_c5_varied, encoding='gb18030')
    # ## start of fill na, cut
    # printlog('class 5 - refreshing varied feature')
    # fe_c5_t = Preprocess.pattern_to_feature(ds_c5_varied, fe_c5_pattern, encoding='gb18030')[0]
    # Preprocess.fill_na(ds_c5_varied, fe_c5_t, replacement=-1, save_path=ds_c5_na, encoding='gb18030')
    # Temp_support.cut(ds_c5_na, fe_c5_t, method='equal-frequency', threshold=10, bin=10, save_path=ds_c5_cut1, encoding='gb18030')
    # printlog('-----------------------------------class 3 - ir-----------------------------------')
    # ## extract class and label feature
    # ds_t = pd.read_csv(ds_path, encoding='gb18030', header=0, index_col=0)
    # pd.concat([ds_t.loc[:, fe_c3], ds_t.iloc[:, -1]], axis=1).to_csv(ds_c3, encoding='gb18030')
    # ## start of fill na, cut
    # Preprocess.fill_na(ds_c3, fe_c3, replacement=-1, save_path=ds_c3_na, encoding='gb18030')
    # Temp_support.cut(ds_c3_na, fe_c3, method='equal-frequency', threshold=5, bin=5,   save_path=ds_c3_cut1, encoding='gb18030')
    # printlog('-----------------------------------class 6 - cf-----------------------------------')
    # ## extract class and label features
    # ds_t = pd.read_csv(ds_path, encoding='gb18030', header=0, index_col=0)
    # pd.concat([ds_t.loc[:, fe_c6], ds_t.iloc[:, -1]], axis=1).to_csv(ds_c6, encoding='gb18030')
    # ## start of fill na, cut
    # Preprocess.fill_na(ds_c6, fe_c6, replacement=-1, save_path=ds_c6_na, encoding='gb18030')
    # Temp_support.cut(ds_c6_na, fe_c6, method='equal-frequency', threshold=5, bin=5,   save_path=ds_c6_cut1, encoding='gb18030')
    # printlog('-----------------------------------class 7 - cons-----------------------------------')
    # ## extract class and flag features
    # ds_t = pd.read_csv(ds_path, encoding='gb18030', header=0, index_col=0)
    # pd.concat([ds_t.loc[:, fe_c7], ds_t.iloc[:, -1]], axis=1).to_csv(ds_c7, encoding='gb18030')
    # ## start of fill na, cut
    # Preprocess.fill_cat(ds_c7, fe_c7, save_path=ds_c7_cat, encoding='gb18030')
    # Preprocess.fill_na(ds_c7_cat, fe_c7, replacement=-1, save_path=ds_c7_na, encoding='gb18030')
    # Temp_support.cut(ds_c7_na, fe_c7, method='equal-frequency', threshold=5, bin=5, save_path=ds_c7_cut1, encoding='gb18030')
    # printlog('-----------------------------------class 8 - pop-----------------------------------')
    # ## extract class and flag features
    # ds_t = pd.read_csv(ds_c8, encoding='gb18030', header=0, index_col=0)
    # ds_origin_t = pd.read_csv(ds_path, encoding='gb18030', header=0, index_col=0)
    # pd.concat([ds_t.loc[:, fe_c8], ds_origin_t.iloc[:, -1]], axis=1, sort=True).to_csv(ds_c8, encoding='gb18030')
    # ## start of selection
    # Preprocess.fill_na(ds_c8, fe_c8, replacement=-1, save_path=ds_c8_na, encoding='gb18030')
    # Temp_support.cut(ds_c8_na, fe_c8, method='equal-frequency', threshold=5, bin=5, save_path=ds_c8_cut1, encoding='gb18030')


    ## necessary as new features in ds_c5_varied
    fe_c5_t = Preprocess.pattern_to_feature(ds_c5_varied, fe_c5_pattern, encoding='gb18030')[0]


    # printlog('-----------------------------------feature selection on IV-----------------------------------')
    # for ds_cut1, fe_iv, ivv in zip(classed_ds_cut1, classed_fe_iv, classed_iv):
    #     Temp_support.select_feature_iv(ds_cut1, pd.read_csv(ds_cut1, header=0, index_col=0).columns[:-1], -1, 
    #         iv_upper_thresh, iv_lower_thresh, to_file=ivv, encoding='gb18030')
    #     ds_t = pd.read_csv(ivv, header=0, index_col=0)['iv']
    #     ds_t[ds_t.between(iv_lower_thresh, iv_upper_thresh)].to_csv(fe_iv, header=0)
    # printlog('-----------------------------------feature selection on lasso-----------------------------------')
    # for i, (ds_na, fe_lasso, lass) in tqdm(enumerate(zip(classed_ds_na, classed_fe_lasso, classed_lasso)), desc='lasso', total=5):
        # # if i == 1 or i == 0:
        # #     continue
        # # alphas = [1, 2, 3, 4, 5]
        # # ds_t = pd.read_csv(ds_na, encoding='gb18030', header=0, index_col=0)
        # # cv_lasso = [np.sqrt(-cross_val_score(Lasso(alpha = alpha), ds_t.iloc[:, :-1], ds_t.iloc[:, -1], 
        # #     scoring="neg_mean_squared_error", cv = 5)).mean() for alpha in alphas]
        # # cv_lasso = pd.Series(cv_lasso, index = alphas)
        # # cv_lasso.plot(title = "Validation")
        # # plt.xlabel("alpha")
        # # plt.ylabel("rmse")
        # # plt.savefig('lasso/lasso_rmse_{}.png'.format(i + 1))
        # # plt.close()
    #     lasso = Lasso(alpha=lasso_alpha)
    #     ds_t = pd.read_csv(ds_na, encoding='gb18030', header=0, index_col=0)
    #     lasso.fit(ds_t.iloc[:, :-1].values, ds_t.iloc[:, -1].values)
    #     pd.DataFrame(lasso.coef_, index=ds_t.columns[:-1], columns=['lasso']).to_csv(lass)
    #     pd.read_csv(lass, header=0, index_col=0).loc[np.abs(lasso.coef_) > lasso_coef, :].to_csv(fe_lasso)
    # printlog('-----------------------------------feature selection on xgb-----------------------------------')
    # for ds_na, fe_xgb, xgbb in zip(classed_ds_na, classed_fe_xgb, classed_xgb):
    #     xgb_t = XGBClassifier()
    #     ds_t = pd.read_csv(ds_na, encoding='gb18030', header=0, index_col=0)
    #     xgb_t.fit(ds_t.iloc[:, :-1].values, ds_t.iloc[:, -1].values)
    #     pd.DataFrame(xgb_t.feature_importances_, index=ds_t.columns[:-1], columns=['xgb']).to_csv(xgbb)
    #     pd.read_csv(xgbb, header=0, index_col=0).iloc[xgb_t.feature_importances_.argsort()[-30:], :].to_csv(fe_xgb)

    # ################### extract top 30 features by iv from every cutting and subclasses ###################
    # printlog('feature selection on IV')
    # from_folder, to_folder = 'iv/', 'features/'
    # classed_ds_iv = [
    #     'ds_c3_ir_iv_cut_1.csv', 'ds_c5_als_varied_iv_cut_1.csv', 'ds_c6_ir_iv_cut_1.csv', 
    #     'ds_c3_ir_iv_cut_2.csv', 'ds_c5_als_varied_iv_cut_2.csv', 'ds_c6_ir_iv_cut_2.csv', 
    #     'ds_c7_ir_iv_cut_1.csv', 'ds_c8_ir_iv_cut_1.csv',
    #     'ds_c7_ir_iv_cut_2.csv', 'ds_c8_ir_iv_cut_2.csv'
    # ]
    # classed_log_iv = [
    #     'fe_c3_ir_iv_1.csv', 'fe_c5_als_iv_1.csv', 'fe_c6_ir_iv_1.csv', 
    #     'fe_c3_ir_iv_2.csv', 'fe_c5_als_iv_2.csv', 'fe_c6_ir_iv_2.csv', 
    #     'fe_c7_ir_iv_1.csv', 'fe_c8_ir_iv_1.csv',
    #     'fe_c7_ir_iv_2.csv', 'fe_c8_ir_iv_2.csv'
    # ]
    # ## extract top 30 features
    # for i, (ds_iv, log_iv) in enumerate(zip(classed_ds_iv, classed_log_iv)):
    #     ds_t = pd.read_csv(from_folder + ds_iv, header=0, index_col=0)
    #     ds_t.sort_values(by='iv', ascending=False).head(30).to_csv(to_folder + log_iv)
    # ## find features in common
    # top_features = []
    # for log_iv in classed_log_iv:
    #     top_features.extend(pd.read_csv(to_folder + log_iv, header=0, index_col=0).index.tolist())
    # top_features = [(str)(key) for key, value in Counter(top_features).items() if value == 2]
    # print(top_features)
    # top_features_row = []
    # for log_iv in classed_log_iv:
    #     for feature in top_features:
    #         if feature in pd.read_csv(to_folder + log_iv, header=0, index_col=0).index:
    #             top_features_row.append(pd.read_csv(to_folder + log_iv, header=0, index_col=0).loc[feature, :])
    #             top_features_row[-1] = pd.DataFrame(top_features_row[-1].values, index=[feature], columns=['iv'])
    # pd.concat(top_features_row).to_csv('features/common_top_features.csv')
    # ## plotting the performance of top common features in cut 1 and cut 2
    # ds_t = pd.read_csv('features/common_top_features.csv', header=0, index_col=0)
    # cut_1_features = ds_t.iloc[0:(int)(ds_t.shape[0] / 2), :].values.tolist()
    # cut_2_features = ds_t.iloc[(int)(ds_t.shape[0] / 2): , :].values.tolist()
    # cut_1_features = [feature[0] for feature in cut_1_features]
    # cut_2_features = [feature[0] for feature in cut_2_features]
    # plt.hist([cut_1_features, cut_2_features], bins=20, label=['分箱数: 5', '分箱数: 10'])
    # plt.xlabel('字段')
    # plt.ylabel('IV值')
    # plt.title('不同分箱中入模变量IV值表现情况')
    # plt.legend()
    # plt.savefig('misc/all_top30_common_iv.png')
    # plt.close()




    # ##################### visualization class 5 - als #####################
    # printlog('visualization')
    # ds_t = pd.read_csv(ds_c5, encoding='gb18030', header=0, index_col=0)
    # for i, prefix_class in enumerate(als_preffix):
    #     df_list = []
    #     for p in prefix_class:
    #         features = Preprocess.pattern_to_feature(ds_t, p, encoding='gb18030')[0]
    #         printlog('doing {}'.format(p))
    #         for feature in features:
    #             count_all   = ds_t.shape[0]
    #             count_na    = ds_t.loc[:, feature].isna().sum()
    #             count_notna = ds_t.loc[:, feature].notna().sum()
    #             count_value = len(list(set(np.ravel(ds_t.loc[ds_t[feature].notna(), feature].values))))
    #             df_list.append(pd.DataFrame([count_all, count_na, count_notna, count_value], 
    #                 columns=[feature], index=['count_all', 'count_na', 'count_notna', 'count_value']))
    #             # printlog(df_list[-1])
    #     pd.concat(df_list, axis=1).to_csv('misc/als_{}.csv'.format(i))

    # ds_t_1 = pd.read_csv('misc/als_0.csv', header=0, index_col=0)
    # ds_t_2 = pd.read_csv('misc/als_1.csv', header=0, index_col=0)
    # ## value
    # printlog('visualization - value')
    # plt.hist([ds_t_1.loc['count_value', :], ds_t_2.loc['count_value', :]], color=['r', 'b'], 
    #     alpha=0.3, label=['id子字段', 'cell子字段'])
    # plt.xlabel('字段非空去重值数量')
    # plt.ylabel('字段计数')
    # plt.title('共债信息子字段取值情况')
    # plt.legend()
    # plt.savefig('misc/als_count_value.png')
    # plt.close()
    # ## na
    # printlog('visualization - na')
    # plt.hist([ds_t_1.loc['count_na', :], ds_t_2.loc['count_na', :]], color=['r', 'b'],
    #     alpha=0.3, label=['id子字段', 'cell子字段'])
    # plt.xlabel('字段空值数量')
    # plt.ylabel('字段计数')
    # plt.title('共债信息子字段空值情况')
    # plt.legend()
    # plt.savefig('misc/als_count_na.png')
    # plt.close()
    # ## iv
    # printlog('visualization - iv')
    # ds_t_1 = pd.read_csv('iv/ds_c5_als_varied_iv_cut_1_adjusted.csv', header=0, index_col=0)
    # ds_t_2 = pd.read_csv('iv/ds_c5_als_varied_iv_cut_2_adjusted.csv', header=0, index_col=0)
    # for i, ds_t in enumerate([ds_t_1, ds_t_2]):
    #     fe_large   = Preprocess.index_pattern_to_feature(ds_t_1, '.*_large_.*')[0]
    #     fe_small   = Preprocess.index_pattern_to_feature(ds_t_1, '.*_small_.*')[0]
    #     fe_id      = Preprocess.index_pattern_to_feature(ds_t_1, '.*_id_.*')[0]
    #     fe_cell    = Preprocess.index_pattern_to_feature(ds_t_1, '.*_cell_.*')[0]
    #     sns.kdeplot(ds_t_1.loc[fe_large, :].T.values[0], label='large衍生字段', shade=False, legend=False, color='r', alpha=0.6)
    #     sns.kdeplot(ds_t_1.loc[fe_small, :].T.values[0], label='small衍生字段', shade=False, legend=False, color='g', alpha=0.6)
    #     sns.kdeplot(ds_t_1.loc[fe_id,    :].T.values[0], label='id衍生字段',    shade=False, legend=False, color='b', alpha=0.6)
    #     sns.kdeplot(ds_t_1.loc[fe_cell,  :].T.values[0], label='cell衍生字段',  shade=False, legend=False, color='m', alpha=0.6)
    #     plt.xlabel('字段IV值')
    #     plt.ylabel('字段计数')
    #     plt.title('共债信息衍生字段IV值表现情况 - 分箱 {}'.format(i + 1))
    #     plt.legend()
    #     plt.savefig('misc/als_iv_cut_{}.png'.format(i + 1))
    #     plt.close()

    #     sns.distplot(ds_t_1.loc[fe_large, :].T.values[0], kde=True, bins=20, rug=True, label='large衍生字段')
    #     table_data=[
    #         ['(0.00, 0.02]',   (int)((ds_t_1.loc[fe_large, :] <= 0.02).sum())],
    #         ['(0.02, 0.10]', (int)(((ds_t_1.loc[fe_large, :] > 0.02) & (ds_t_1.loc[fe_large, :] <= 0.1)).values.sum())],
    #         ['(0.10, 0.30]',  (int)(((ds_t_1.loc[fe_large, :] > 0.1)  & (ds_t_1.loc[fe_large, :] <= 0.3)).values.sum())],
    #         ['(0.30, 0.50]',  (int)(((ds_t_1.loc[fe_large, :] > 0.3)  & (ds_t_1.loc[fe_large, :] <= 0.5)).values.sum())],
    #         ['> 0.50',        (int)((ds_t_1.loc[fe_large, :] > 0.5).sum())]
    #     ]
    #     table = plt.table(cellText=table_data, loc='bottom', cellLoc='left', 
    #         bbox=[1.05, 0.6, 0.4, 0.4], colWidths=[0.27, 0.13])
    #     table.auto_set_font_size(False)
    #     table.set_fontsize(10)
    #     plt.subplots_adjust(right=0.7)
    #     plt.xlabel('字段IV值')
    #     plt.ylabel('字段计数')
    #     plt.xlim(right=0.65)
    #     plt.ylim(top=10)
    #     plt.title('共债信息衍生字段IV值表现情况 - 按值衍生(较大值) + 分箱 {}'.format(i + 1))
    #     plt.legend()
    #     plt.savefig('misc/als_iv_cut_{}_large.png'.format(i + 1))
    #     plt.close()

    #     sns.distplot(ds_t_1.loc[fe_small, :].T.values[0], kde=True, bins=20, rug=True, label='small衍生字段')
    #     table_data=[
    #         ['(0.00, 0.02]',   (int)((ds_t_1.loc[fe_small, :] <= 0.02).sum())],
    #         ['(0.02, 0.10]', (int)(((ds_t_1.loc[fe_small, :] > 0.02) & (ds_t_1.loc[fe_small, :] <= 0.1)).values.sum())],
    #         ['(0.10, 0.30]',  (int)(((ds_t_1.loc[fe_small, :] > 0.1)  & (ds_t_1.loc[fe_small, :] <= 0.3)).values.sum())],
    #         ['(0.30, 0.50]',  (int)(((ds_t_1.loc[fe_small, :] > 0.3)  & (ds_t_1.loc[fe_small, :] <= 0.5)).values.sum())],
    #         ['> 0.50',        (int)((ds_t_1.loc[fe_small, :] > 0.5).sum())]
    #     ]
    #     table = plt.table(cellText=table_data, loc='bottom', cellLoc='left', 
    #         bbox=[1.05, 0.6, 0.4, 0.4], colWidths=[0.27, 0.13])
    #     table.auto_set_font_size(False)
    #     table.set_fontsize(10)
    #     plt.subplots_adjust(right=0.7)
    #     plt.xlabel('字段IV值')
    #     plt.ylabel('字段计数')
    #     plt.xlim(right=0.65)
    #     plt.ylim(top=10)
    #     plt.title('共债信息衍生字段IV值表现情况 - 按值衍生(较小值) + 分箱 {}'.format(i + 1))
    #     plt.legend()
    #     plt.savefig('misc/als_iv_cut_{}_small.png'.format(i + 1))
    #     plt.close()

    #     sns.distplot(ds_t_1.loc[fe_id, :].T.values[0], kde=True, bins=20, rug=True, label='id衍生字段')
    #     table_data=[
    #         ['(0.00, 0.02]',   (int)((ds_t_1.loc[fe_id, :] <= 0.02).sum())],
    #         ['(0.02, 0.10]', (int)(((ds_t_1.loc[fe_id, :] > 0.02) & (ds_t_1.loc[fe_id, :] <= 0.1)).values.sum())],
    #         ['(0.10, 0.30]',  (int)(((ds_t_1.loc[fe_id, :] > 0.1)  & (ds_t_1.loc[fe_id, :] <= 0.3)).values.sum())],
    #         ['(0.30, 0.50]',  (int)(((ds_t_1.loc[fe_id, :] > 0.3)  & (ds_t_1.loc[fe_id, :] <= 0.5)).values.sum())],
    #         ['> 0.50',        (int)((ds_t_1.loc[fe_id, :] > 0.5).sum())]
    #     ]
    #     table = plt.table(cellText=table_data, loc='bottom', cellLoc='left', 
    #         bbox=[1.05, 0.6, 0.4, 0.4], colWidths=[0.27, 0.13])
    #     table.auto_set_font_size(False)
    #     table.set_fontsize(10)
    #     plt.subplots_adjust(right=0.7)
    #     plt.xlabel('字段IV值')
    #     plt.ylabel('字段计数')
    #     plt.xlim(right=0.65)
    #     plt.ylim(top=10)
    #     plt.title('共债信息衍生字段IV值表现情况 - 按字段衍生(id) + 分箱 {}'.format(i + 1))
    #     plt.legend()
    #     plt.savefig('misc/als_iv_cut_{}_id.png'.format(i + 1))
    #     plt.close()

    #     sns.distplot(ds_t_1.loc[fe_cell, :].T.values[0], kde=True, bins=20, rug=True, label='cell衍生字段')
    #     table_data=[
    #         ['(0.00, 0.02]',   (int)((ds_t_1.loc[fe_cell, :] <= 0.02).sum())],
    #         ['(0.02, 0.10]', (int)(((ds_t_1.loc[fe_cell, :] > 0.02) & (ds_t_1.loc[fe_cell, :] <= 0.1)).values.sum())],
    #         ['(0.10, 0.30]',  (int)(((ds_t_1.loc[fe_cell, :] > 0.1)  & (ds_t_1.loc[fe_cell, :] <= 0.3)).values.sum())],
    #         ['(0.30, 0.50]',  (int)(((ds_t_1.loc[fe_cell, :] > 0.3)  & (ds_t_1.loc[fe_cell, :] <= 0.5)).values.sum())],
    #         ['> 0.50',        (int)((ds_t_1.loc[fe_cell, :] > 0.5).sum())]
    #     ]
    #     table = plt.table(cellText=table_data, loc='bottom', cellLoc='left', 
    #         bbox=[1.05, 0.6, 0.4, 0.4], colWidths=[0.27, 0.13])
    #     table.auto_set_font_size(False)
    #     table.set_fontsize(10)
    #     plt.subplots_adjust(right=0.7)
    #     plt.xlabel('字段IV值')
    #     plt.ylabel('字段计数')
    #     plt.xlim(right=0.65)
    #     plt.ylim(top=10)
    #     plt.title('共债信息衍生字段IV值表现情况 - 按字段衍生(cell) + 分箱 {}'.format(i + 1))
    #     plt.legend()
    #     plt.savefig('misc/als_iv_cut_{}_cell.png'.format(i + 1))
    #     plt.close()
    

    # printlog('-----------------------------------experience feature-----------------------------------')
    # series_t = pd.read_csv(ds_c7_na, encoding='gb18030', header=0, index_col=0)['cons_tot_m12_visits']
    # series_t[series_t.between(-99.001, -0.001)]    = -99
    # series_t[series_t.between(-0.001, 500.001)]    = 500
    # series_t[series_t.between(500.001, 1000.001)]  = 1000
    # series_t[series_t.between(1000.001, 1500.001)] = 1500
    # series_t[series_t.between(1500.001, 900000)]   = 9000
    # series_t.to_csv(ds_exp_na, header='cons_tot_m12_visits')

    # ds_t = pd.read_csv(ds_c8_na, encoding='gb18030', header=0, index_col=0)[['pd_id_gender', 'pd_id_apply_age']]
    # series_t = pd.Series(data=-1, index=ds_t.index)
    # series_t[(ds_t['pd_id_gender'] == 0) & (ds_t['pd_id_apply_age'].between(-99.001, 30.001))] = 0
    # series_t[(ds_t['pd_id_gender'] == 0) & (ds_t['pd_id_apply_age'].between(30.001, 60.001))]  = 1
    # series_t[(ds_t['pd_id_gender'] == 0) & (ds_t['pd_id_apply_age'].between(60.001, 999.001))] = 2
    # series_t[(ds_t['pd_id_gender'] == 1) & (ds_t['pd_id_apply_age'].between(-0.001, 24.001))]  = 3
    # series_t[(ds_t['pd_id_gender'] == 1) & (ds_t['pd_id_apply_age'].between(24.001, 35.001))]  = 4
    # series_t[(ds_t['pd_id_gender'] == 1) & (ds_t['pd_id_apply_age'].between(35.001, 45.001))]  = 5
    # series_t[(ds_t['pd_id_gender'] == 1) & (ds_t['pd_id_apply_age'].between(45.001, 999.001))] = 2
    # pd.concat([pd.DataFrame(series_t, columns=['pd_gender_age']), pd.read_csv(ds_exp_na, encoding='gb18030', 
    #     header=0, index_col=0)], axis=1, sort=True).to_csv(ds_exp_na, encoding='gb18030')

    
    printlog('-----------------------------------prepare dataset-----------------------------------')
    hitrate_features = Log.iterread(fe_gate_hit)
    tree_features    = Log.iterread(fe_gate_tree)
    xgb_features     = [
        'als_m12_id_nbank_orgnum', 'als_m3_id_cooff_allnum',
        'ir_id_x_cell_cnt', 'cons_tot_m12_visits',
        'als_m6_id_rel_allnum', 'als_fst_id_nbank_inteday']
    pop_features     = []
    exp_features     = ['cons_tot_m12_visits', 'pd_gender_age']
    ds_model         = 'tmp/ds_model.csv'
    ds_model_cat     = 'tmp/ds_model_cat.csv'
    ds_model_na      = 'tmp/ds_model_na.csv'
    ds_t = pd.read_csv(ds_path, encoding='gb18030', header=0, index_col=0)
    pop_t = pd.read_csv(ds_c8, encoding='gb18030', header=0, index_col=0)
    ds_exp_t = pd.read_csv(ds_exp_na, encoding='gb18030', header=0, index_col=0)
    pd.concat([ds_exp_t.loc[:, exp_features], ds_t.loc[:, hitrate_features+tree_features+xgb_features], pop_t.loc[:, pop_features], ds_t.iloc[:, -1]], axis=1, sort=True).to_csv(ds_model)
    Preprocess.fill_na(ds_model, hitrate_features+tree_features+xgb_features+pop_features, replacement=-1, save_path=ds_model_na, encoding='gb18030')
    Preprocess.fill_cat(ds_model_na, hitrate_features+tree_features+xgb_features+pop_features, save_path=ds_model_cat, encoding='gb18030')
    ds_t = pd.read_csv(ds_model_cat, header=0, index_col=0, encoding='gb18030')
    train_fe, test_fe, train_lb, test_lb = train_test_split(ds_t.iloc[:, :-1], ds_t.iloc[:, -1], test_size=0.3, train_size=0.7, random_state=1)
    printlog('train label proportion: {}'.format(train_lb.sum() / train_lb.count()))
    printlog('test label proportion: {}'.format(test_lb.sum() / test_lb.count()))
    printlog('-----------------------------------gate and tree-----------------------------------')
    pred_hit     = (test_fe[hitrate_features] != -1).any(axis=1).astype(int)
    pred_tree    = pd.Series(load(tree_gate).predict(test_fe[tree_features]), index=test_fe.index)
    printlog('gate test: {} labelled 1 by hit positive rate.'.format(pred_hit.sum()))
    printlog('gate test: {} labelled 1 by tree classifier.'.format(pred_tree.sum()))
    printlog('-----------------------------------train on xgb-----------------------------------')
    def objective(y_true, y_pred):
        multiplier = pd.Series(y_true).mask(y_true == 1, xgb_FP_grad_mul).mask(y_true == 0, xgb_FN_grad_mul)
        grad = multiplier * (y_pred - y_true)
        hess = np.power(np.abs(grad), 0.5)
        return grad, hess
    xgb_t = XGBClassifier(objective=objective)
    xgb_t.fit(train_fe, train_lb)
    prediction = xgb_t.predict_proba(test_fe)
    ## apply gate prediction to xgb prediction
    prediction[:, 1] += pred_hit + pred_tree
    prediction[prediction[:, 1] > 1, 1] = 1
    printlog(prediction)
    Assess.xgbAssess(test_lb.to_numpy(), prediction, 'misc')

















    # ############################ LR #########################
    # from sklearn.linear_model import LogisticRegression
    # clf = LogisticRegression()
    # clf.fit(train_fe, train_lb)
    # prediction = clf.predict_proba(test_fe).tolist()
    # for i, pre in enumerate(prediction):
    #     prediction[i] = pre[1]
    # plt.scatter(prediction, test_lb, s=0.3, label='测试集表现')
    # plt.title('Logistic预测表现')
    # plt.xlabel('Logistic预测值分布')
    # plt.ylabel('测试集标签值分布')
    # plt.legend()
    # plt.savefig('misc/lr_1.png')
    # plt.close()
    # sns.distplot(prediction, bins=15, label='Logistic预测值')
    # sns.distplot(test_lb,    bins=15, label='测试集标签值')
    # plt.title('Logistic预测表现KDE-直方图')
    # plt.xlabel('标签/预测值')
    # plt.ylabel('标签/预测值分布')
    # plt.legend()
    # plt.savefig('misc/lr.png')
    # plt.close()

    # from sklearn.linear_model import LogisticRegression
    # clf = LogisticRegression(random_state=0).fit(train_fe.values, train_lb.values)
    # cnt_crt, cnt = 0, 0
    # prediction = clf.predict(test_fe.values)
    # for predict, truth in zip(prediction, test_lb.values):
    #     if predict == truth:
    #         cnt_crt += 1
    #     cnt += 1
    # printlog('test correction rate: {}'.format(cnt_crt / cnt))
    





    # gate_feature = ds.iloc[ds[[feature]].notna(), -1] for feature in mut_exc_feature
    # new_class_1_features = []
    # print(len(list(itertools.chain.from_iterable(classed_class_1_features))))
    # for feature_class in classed_class_1_features:
    #     new_class_1_features.append()


    

if __name__ == '__main__':
    run()
    potplayer.run('jinitaimei.m4a')