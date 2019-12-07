from utils import EDA_massive, Preprocess, Log, EDA, Feature_selection, Model, Temp_support
from utils.Log import printlog

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from collections import Counter
import matplotlib.pyplot as plt
from functools import reduce
from sklearn import tree
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import winsound

def run():
    ds_path = 'data/data.csv'
    # ds_smp_path = 'tmp/ds_smp_1.csv'
    feature_selection_log = 'logs/feature_selection.log'
    # ds_smp_spe_path = 'tmp/ds_smp_spe.csv'
    # ds_smp_srt_path = 'tmp/ds_smp_srt.csv'
    flag_list = ['flag_specialList_c', 'flag_fraudrelation_g', 'flag_inforelation', 'flag_applyloanusury', 'flag_applyloanstr', 'flag_ConsumptionFeature', 'flag_consumption_c']
    check_feature_pattern = ['^sl_', '^frg_', '^ir_', '^alu_', '^als_', '^cf_', '^cons_']
    sample_datasets = [
        'tmp/ds_feature_train.csv',
        'tmp/ds_label_train.csv',
        'tmp/ds_feature_test.csv',
        'tmp/ds_label_test.csv'
    ]

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'SimHei'
    Log.clear_log(creative=True)
    Log.clear_log(file_path=feature_selection_log, creative=True)
    # #################### necessary for afterward debugging #####################
    # classed_features = Preprocess.pattern_to_feature(ds_smp_path, check_feature_pattern)
    # labels = EDA_massive.labels(ds_smp_srt_path, column=-1, encoding='gb18030')
    # classed_preffix = [Temp_support.prefix_from_meta(fl) for fl in flag_list]




    # #************************ preprocess ************************
    # ##################### data cleaning #####################
    # ## unnecessary; if data cleaning is performed, the proceeding fill_cat will not work
    # special_features = reduce(lambda accum, fc: np.concatenate([accum, Preprocess.special_feature(ds_smp_path, fc)]), classed_features, np.array([]))
    # Preprocess.pop_feature(ds_smp_path, special_features, save_path=ds_smp_path, pop_path=ds_smp_spe_path, encoding='gb18030')


    # #################### feature selection #####################
    # #################### class 1 - sl      #####################
    # printlog('class 1 - sl')
    # ## mut_exc_1_feature = Temp_support.feature_padding(ds_smp_path, classed_features[0], classed_preffix[0], encoding='gb18030')
    # mut_exc_1_feature = Temp_support.feature_padding_on_hit_rate(ds_smp_path, classed_features[0], classed_preffix[0], encoding='gb18030')
    # printlog('class 1 - original features: {}'.format(mut_exc_1_feature), printable=False)
    # ## the class_1_gate_feature is essential for gateway classification
    # class_1_gate_feature = Feature_selection.hit_positive_rate(ds_smp_path, mut_exc_1_feature, -1, 0.5, encoding='gb18030')
    # printlog('class 1 - gate feature: {}'.format(class_1_gate_feature), file_path=feature_selection_log, printable=False)
    # tocheck_feature = Feature_selection.hit_positive_rate(ds_smp_path, mut_exc_1_feature, -1, 0.1, encoding='gb18030')
    # tocheck_feature = [feature for feature in tocheck_feature if feature not in class_1_gate_feature]
    # printlog('class 1 - tocheck feature: {}'.format(tocheck_feature), file_path=feature_selection_log, printable=False)
    # ## if the features are categorical features, put them into decision tree and derive the tree model
    # fill_na = lambda x, f: Preprocess.fill_na(x, f, flag_feature=flag_list[0], save_path=ds_smp_path, flag_replacement=-1, encoding='gb18030')
    # ## the class_1_tree is essential for tree classifier here
    # if len(tocheck_feature) > 0:
    #     class_1_tree = Model.tree_classifier(
    #         ds=ds_smp_path, 
    #         features=tocheck_feature, 
    #         label_column=-1, 
    #         fill_na=fill_na, 
    #         encoding='gb18030', 
    #         export_path='tmp/class_1_tree.dot'
    #     )
    # else:
    #     printlog('class 1 - tocheck feature: no feature to check with tree')
    # ##################### class 2 - fr #####################
    # ## note that classed_features[1] is updated here for label binarizer, new ds is saved at save_path in fill_cat
    # printlog('class 2 - fr')
    # fill_na = lambda x, f: Preprocess.fill_na(x, f, flag_feature=flag_list[1], flag_replacement=-1, encoding='gb18030')
    # fill_cat = lambda x, f: Preprocess.fill_cat(x, f, method='label_binarizer', save_path=ds_smp_path, encoding='gb18030')
    # ## the class_2_tree, class_2_categorical_encoder, classed_feature[1] are essential for tree classifier here
    # class_2_tree, class_2_categorical_encoder, classed_features[1] = Model.tree_classifier(
    #     ds=ds_smp_path, 
    #     features=classed_features[1], 
    #     label_column=-1, 
    #     fill_na=fill_na, 
    #     fill_cat=fill_cat, 
    #     export_path='tmp/class_2_tree.dot', 
    #     encoding='gb18030'
    # )
    # ## 2-layer-tree sprouting on feature
    # Temp_support.two_layer_tree(ds_smp_path, classed_features[1][0], 
    #     classed_features[1][1], -1, to_file='tmp/class_2_tree_new.dot', encoding='gb18030')
    # Temp_support.two_layer_tree(ds_smp_path, classed_features[1][1], 
    #     classed_features[1][0], -1, to_file='tmp/class_2_tree_new_t.dot', encoding='gb18030')
    # ##################### class 4 - alu #####################
    # printlog('class 4 - alu')
    # mut_exc_4_feature = classed_features[3]
    # printlog('class 4 - original features: {}'.format(mut_exc_4_feature), printable=False)
    # ## the class_4_gate_feature is essential for gateway classification
    # class_4_gate_feature = Feature_selection.hit_positive_rate(ds_smp_path, mut_exc_4_feature, -1, 0.9, encoding='gb18030')
    # printlog('class 4 - gate feature: {}'.format(class_4_gate_feature), file_path=feature_selection_log, printable=False)
    # tocheck_feature = Feature_selection.hit_positive_rate(ds_smp_path, mut_exc_4_feature, -1, 0.6, encoding='gb18030')
    # tocheck_feature = [feature for feature in tocheck_feature if feature not in class_4_gate_feature]
    # printlog('class 4 - tocheck feature: {}'.format(tocheck_feature), file_path=feature_selection_log, printable=False)
    # ## if the features are categorical features, put them into decision tree and derive the tree model
    # fill_na = lambda x, f: Preprocess.fill_na(x, f, flag_feature=flag_list[3], save_path=ds_smp_path, flag_replacement=-1, encoding='gb18030')
    # ## the class_4_tree is essential for tree classifier here
    # if len(tocheck_feature) > 0:
    #     class_4_tree = Model.tree_classifier(
    #         ds=ds_smp_path, 
    #         features=tocheck_feature, 
    #         label_column=-1, 
    #         fill_na=fill_na, 
    #         encoding='gb18030', 
    #         export_path='tmp/class_4_tree.dot'
    #     )
    # else:
    #     printlog('class 4 - tocheck feature: no feature to check with tree')
    # ##################### class 5 - als #####################
    # printlog('class 5 - als')
    # ## class 5 variables
    # ds_c5                 = 'tmp/ds_c5_als.csv'
    # ds_c5_varied          = 'tmp/ds_c5_als_varied.csv'
    # ds_c5_varied_na       = 'tmp/ds_c5_als_varied_na.csv'
    # ds_c5_varied_cut_1 = 'tmp/ds_c5_als_varied_cut_1.csv'
    # ds_c5_varied_cut_2 = 'tmp/ds_c5_als_varied_cut_2.csv'
    # ds_c5_varied_cut_1_iv = 'iv/ds_c5_als_varied_iv_cut_1.csv'
    # ds_c5_varied_cut_2_iv = 'iv/ds_c5_als_varied_iv_cut_2.csv'
    # fe_c5_pattern   = '^als_'
    # fe_c5_cut1_iv  = 'features/fe_c5_als_iv_1.log'
    # fe_c5_cut2_iv  = 'features/fe_c5_als_iv_2.log'
    # als_preffix = [
    #     ['^als_d7_id_', '^als_d15_id_', '^als_m1_id_', '^als_m3_id_', '^als_m6_id_', '^als_m12_id_', '^als_fst_id_', '^als_lst_id_'],
    #     ['^als_d7_cell_', '^als_d15_cell_', '^als_m1_cell_', '^als_m3_cell_', '^als_m6_cell_', '^als_m12_cell_', '^als_fst_cell_', '^als_lst_cell_']]
    # ## class 5 ds
    # fe_c5           = Preprocess.pattern_to_feature(ds_path, fe_c5_pattern, encoding='gb18030')[0]
    # ds_t = pd.read_csv(ds_path, encoding='gb18030', header=0, index_col=0)
    # pd.concat([ds_t.loc[:, fe_c5], ds_t.iloc[:, -1]], axis=1).to_csv(ds_c5, encoding='gb18030')
    # ## start of variation
    # printlog('class 5 - value padding: larger/smaller')
    # ds_t = pd.read_csv(ds_c5, encoding='gb18030', header=0, index_col=0)
    # for i, (id_fc, cell_fc) in enumerate(zip(Preprocess.pattern_to_feature(ds_t, als_preffix[0], encoding='gb18030'), Preprocess.pattern_to_feature(ds_t, als_preffix[1], encoding='gb18030'))):
    #     for id_f, cell_f in zip(id_fc, cell_fc):
    #         large_f = id_f.replace('id', 'large')
    #         small_f = id_f.replace('id', 'small')
    #         ds_t.insert(loc=ds_t.columns.get_loc(id_f), column=large_f, value=np.nan)
    #         ds_t.insert(loc=ds_t.columns.get_loc(id_f), column=small_f, value=np.nan)
    #         for row in ds_t[ds_t[cell_f].notna()].index:
    #             ## change < to > for smaller value padding
    #             ds_t.loc[row, large_f] = ds_t.loc[row, cell_f] if ds_t.loc[row, id_f] == np.nan or ds_t.loc[row, id_f] < ds_t.loc[row, cell_f] else ds_t.loc[row, id_f]
    #             ds_t.loc[row, small_f] = ds_t.loc[row, cell_f] if ds_t.loc[row, id_f] == np.nan or ds_t.loc[row, id_f] > ds_t.loc[row, cell_f] else ds_t.loc[row, id_f]
    # ds_t.to_csv(ds_c5_varied, encoding='gb18030')
    # printlog('class 5 - feature padding: id/cell')
    # pass
    # ## start of cleaning
    # printlog('class 5 - refreshing varied feature')
    # fe_c5_varied = Preprocess.pattern_to_feature(ds_c5_varied, fe_c5_pattern, encoding='gb18030')[0]
    # printlog('class 5 - fill na')
    # Preprocess.fill_na(ds_c5_varied, fe_c5_varied, replacement=-1, save_path=ds_c5_varied_na, encoding='gb18030')
    # printlog('class 5 - cut 1')
    # Temp_support.cut(ds_c5_varied_na, fe_c5_varied, threshold=10, bin=10, method='equal-distance', save_path=ds_c5_varied_cut_1, encoding='gb18030')
    # printlog('class 5 - cut 2')
    # Temp_support.cut(ds_c5_varied_na, fe_c5_varied, threshold=5 , bin=5,  method='equal-distance', save_path=ds_c5_varied_cut_2, encoding='gb18030')
    # printlog('class 5 - select by iv 1')
    # fe_c5_iv_1 = Temp_support.select_feature_iv(ds_c5_varied_cut_1, fe_c5_varied, -1, 0.5, 0.3, to_file=ds_c5_varied_cut_1_iv, encoding='gb18030')
    # printlog('class 5 - select by iv 2')
    # fe_c5_iv_2 = Temp_support.select_feature_iv(ds_c5_varied_cut_2, fe_c5_varied, -1, 0.5, 0.3, to_file=ds_c5_varied_cut_2_iv, encoding='gb18030')
    # printlog('class 5 - saving')
    # Log.itersave(file_path=fe_c5_cut1_iv, iteritem=fe_c5_iv_1)
    # Log.itersave(file_path=fe_c5_cut2_iv, iteritem=fe_c5_iv_2)
    ##################### class 3 - ir #####################
    ''' data flow:
        ds_c[n]                 raw dataset
        ds_c[n]_na              after na data are filled
        ds_c[n]_cut[1/2]        after data are cut by method 1/2
        iv_c[n]_cut[1/2]        dataset of features as rows and IVs as columns
        fe_c[n]_cut[1/2]_iv     dataset of features as rows and IVs as columns
        fe_c[n]_pattern         prefix pattern for class features
        fe_c[n]                 list of class features strings
    '''
    printlog('class 3 - ir')
    ## class 3 variables
    ds_c3         = 'tmp/ds_c3.csv'
    ds_c3_na      = 'tmp/ds_c3_na.csv'
    ds_c3_cut1    = 'tmp/ds_c3_cut1.csv'
    ds_c3_cut2    = 'tmp/ds_c3_cut2.csv'
    iv_c3_cut1    = 'iv/iv_c3_cut1.csv'
    iv_c3_cut2    = 'iv/iv_c3_cut2.csv'
    fe_c3_cut1_iv = 'features/fe_c3_cut1_iv.log'
    fe_c3_cut2_iv = 'features/fe_c3_cut2_iv.log'
    fe_c3_pattern = '^ir_'
    fe_c3         = Preprocess.pattern_to_feature(ds_path, fe_c3_pattern, encoding='gb18030')[0]
    # ## extract class and label feture
    # ds_t = pd.read_csv(ds_path, encoding='gb18030', header=0, index_col=0)
    # pd.concat([ds_t.loc[:, fe_c3], ds_t.iloc[:, -1]], axis=1).to_csv(ds_c3, encoding='gb18030')
    # ## start of selection
    # Preprocess.fill_na(ds_c3, fe_c3, replacement=-1, save_path=ds_c3_na, encoding='gb18030')
    # Temp_support.cut(ds_c3_na, fe_c3, threshold=5, bin=5,   save_path=ds_c3_cut1, encoding='gb18030')
    # Temp_support.cut(ds_c3_na, fe_c3, threshold=10, bin=10, save_path=ds_c3_cut2, encoding='gb18030')
    # Log.itersave(file_path=fe_c3_cut1_iv, iteritem=
    #   Temp_support.select_feature_iv(ds_c3_cut1, fe_c3, -1, 0.5, 0.3, to_file=iv_c3_cut1, encoding='gb18030'))
    # Log.itersave(file_path=fe_c3_cut2_iv, iteritem=
    #   Temp_support.select_feature_iv(ds_c3_cut2, fe_c3, -1, 0.5, 0.3, to_file=iv_c3_cut2, encoding='gb18030'))
    printlog(Temp_support.cut(ds_c3_na, fe_c3, method='optimal', label_column=-1, encoding='gb18030'))
    # ##################### class 6 - cf #####################
    # printlog('class 6 - cf')
    # ## class 6 variables
    # ds_c6          = 'tmp/ds_c6.csv'
    # ds_c6_na       = 'tmp/ds_c6_na.csv'
    # ds_c6_cut1     = 'tmp/ds_c6_cut1.csv'
    # ds_c6_cut2     = 'tmp/ds_c6_cut2.csv'
    # iv_c6_cut1     = 'iv/iv_c6_cut1.csv'
    # iv_c6_cut2     = 'iv/iv_c6_cut2.csv'
    # fe_c6_cut1_iv  = 'features/fe_c6_cut1_iv.log'
    # fe_c6_cut2_iv  = 'features/fe_c6_cut2_iv.log'
    # fe_c6_pattern  = '^cf_'
    # fe_c6       = Preprocess.pattern_to_feature(ds_path, fe_c6_pattern, encoding='gb18030')[0]
    # ## extract class and label features
    # ds_t = pd.read_csv(ds_path, encoding='gb18030', header=0, index_col=0)
    # pd.concat([ds_t.loc[:, fe_c6], ds_t.iloc[:, -1]], axis=1).to_csv(ds_c6, encoding='gb18030')
    # ## start of selection
    # Preprocess.fill_na(ds_c6, fe_c6, replacement=-1, save_path=ds_c6_na, encoding='gb18030')
    # Temp_support.cut(ds_c6_na, fe_c6, threshold=5, bin=5,   save_path=ds_c6_cut1, encoding='gb18030')
    # Temp_support.cut(ds_c6_na, fe_c6, threshold=10, bin=10, save_path=ds_c6_cut2, encoding='gb18030')
    # Log.itersave(file_path=fe_c6_cut1_iv, iteritem=Temp_support.select_feature_iv(ds_c6_cut1, fe_c6, -1, 0.5, 0.3, to_file=iv_c6_cut1, encoding='gb18030'))
    # Log.itersave(file_path=fe_c6_cut2_iv, iteritem=Temp_support.select_feature_iv(ds_c6_cut2, fe_c6, -1, 0.5, 0.3, to_file=iv_c6_cut2, encoding='gb18030'))
    # ##################### class 7 - cons #####################
    # printlog('class 7 - cons')
    # ## class 7 variables
    # ds_c7          = 'tmp/ds_c7.csv'
    # ds_c7_cat      = 'tmp/ds_c7_cat.csv'
    # ds_c7_na       = 'tmp/ds_c7_na.csv'
    # ds_c7_cut1     = 'tmp/ds_c7_cut1.csv'
    # ds_c7_cut2     = 'tmp/ds_c7_cut2.csv'
    # iv_c7_cut1     = 'iv/iv_c7_cut1.csv'
    # iv_c7_cut2     = 'iv/iv_c7_cut2.csv'
    # fe_c7_cut1_iv  = 'features/fe_c7_cut1_iv.log'
    # fe_c7_cut2_iv  = 'features/fe_c7_cut2_iv.log'
    # fe_c7_pattern  = '^cons_'
    # fe_c7          = Preprocess.pattern_to_feature(ds_path, fe_c7_pattern, encoding='gb18030')[0]
    # ## extract class and flag features
    # ds_t = pd.read_csv(ds_path, encoding='gb18030', header=0, index_col=0)
    # pd.concat([ds_t.loc[:, fe_c7], ds_t.iloc[:, -1]], axis=1).to_csv(ds_c7, encoding='gb18030')
    # ## start of selection
    # Preprocess.fill_cat(ds_c7, fe_c7, save_path=ds_c7_cat, encoding='gb18030')
    # Preprocess.fill_na(ds_c7_cat, fe_c7, replacement=-1, save_path=ds_c7_na, encoding='gb18030')
    # Temp_support.cut(ds_c7_na, fe_c7, threshold=5, bin=5,   save_path=ds_c7_cut1, encoding='gb18030')
    # Temp_support.cut(ds_c7_na, fe_c7, threshold=10, bin=10, save_path=ds_c7_cut2, encoding='gb18030')
    # Log.itersave(file_path=fe_c7_cut1_iv, iteritem=
    #     Temp_support.select_feature_iv(ds_c7_cut1, fe_c7, -1, 0.5, 0.3, to_file=iv_c7_cut1, encoding='gb18030'))
    # Log.itersave(file_path=fe_c7_cut2_iv, iteritem=
    #     Temp_support.select_feature_iv(ds_c7_cut2, fe_c7, -1, 0.5, 0.3, to_file=iv_c7_cut2, encoding='gb18030'))
    # ##################### class 8 - pop #####################
    # printlog('class 8 - pop')
    # ## class 8 variables
    # ds_c8          = 'data/pop.csv'
    # ds_c8_na       = 'tmp/ds_c8_na.csv'
    # ds_c8_cut1     = 'tmp/ds_c8_cut1.csv'
    # ds_c8_cut2     = 'tmp/ds_c8_cut2.csv'
    # iv_c8_cut1     = 'iv/iv_c8_cut1.csv'
    # iv_c8_cut2     = 'iv/iv_c8_cut2.csv'
    # fe_c8_cut1_iv  = 'features/fe_c8_cut1_iv.log'
    # fe_c8_cut2_iv  = 'features/fe_c8_cut2_iv.log'
    # fe_c8_pattern  = '^pd_'
    # fe_c8          = Preprocess.pattern_to_feature(ds_c8, fe_c8_pattern, encoding='gb18030')[0]
    # ## extract class and flag features
    # ds_t = pd.read_csv(ds_c8, encoding='gb18030', header=0, index_col=0)
    # ds_origin_t = pd.read_csv(ds_path, encoding='gb18030', header=0, index_col=0)
    # pd.concat([ds_t.loc[:, fe_c8], ds_origin_t.iloc[:, -1]], axis=1, sort=True).to_csv(ds_c8, encoding='gb18030')
    # ## start of selection
    # Preprocess.fill_na(ds_c8, fe_c8, replacement=-1, save_path=ds_c8_na, encoding='gb18030')
    # Temp_support.cut(ds_c8_na, fe_c8, threshold=5, bin=5,   method='equal-frequency', save_path=ds_c8_cut1, encoding='gb18030')
    # Temp_support.cut(ds_c8_na, fe_c8, threshold=10, bin=10, method='equal-frequency', save_path=ds_c8_cut2, encoding='gb18030')
    # Log.itersave(file_path=fe_c8_cut1_iv, iteritem=
    #     Temp_support.select_feature_iv(ds_c8_cut1, fe_c8, -1, 0.5, 0.3, to_file=iv_c8_cut1, encoding='gb18030'))
    # Log.itersave(file_path=fe_c8_cut2_iv, iteritem=
    #     Temp_support.select_feature_iv(ds_c8_cut2, fe_c8, -1, 0.5, 0.3, to_file=iv_c8_cut2, encoding='gb18030'))
    # ####################### feature selection on xgb #######################
    # printlog('feature selection on xgb')
    # classed_ds_na = [
    #     'tmp/ds_c3_ir_na.csv', 'tmp/ds_c5_als_varied_na.csv', 'tmp/ds_c6_ir_na.csv',
    #     'tmp/ds_c7_ir_na.csv', 'tmp/ds_c8_ir_na.csv'
    # ]
    # classed_ds_xgb = [
    #     'xgb/ds_c3_na_xgb.csv', 'xgb/ds_c5_na_xgb.csv', 'xgb/ds_c6_na_xgb.csv',
    #     'xgb/ds_c7_na_xgb.csv', 'xgb/ds_c8_na_xgb.csv'
    # ]
    # classed_ds_feature = [
    #     'features/fe_c3_na_xgb.csv', 'features/fe_c5_na_xgb.csv', 'features/fe_c6_na_xgb.csv',
    #     'features/fe_c7_na_xgb.csv', 'features/fe_c8_na_xgb.csv'
    # ]
    # for ds_na, ds_xgb, ds_feature in zip(classed_ds_na, classed_ds_xgb, classed_ds_feature):
    #     xgb_t = XGBClassifier()
    #     ds_t = pd.read_csv(ds_na, encoding='gb18030', header=0, index_col=0)
    #     xgb_t.fit(ds_t.iloc[:, :-1].values, ds_t.iloc[:, -1].values)
    #     pd.DataFrame(xgb_t.feature_importances_, index=ds_t.columns[:-1], columns=['xgb']).to_csv(ds_xgb)
    #     top_indexing = xgb_t.feature_importances_.argsort()[-30:]
    #     pd.read_csv(ds_xgb, header=0, index_col=0).iloc[top_indexing, :].to_csv(ds_feature)
    # top_features = []
    # for ds_feature in classed_ds_feature:
    #     top_features.append(pd.read_csv(ds_feature, header=0, index_col=0))
    # top_features = pd.concat(top_features, axis=0)
    # top_features.iloc[top_features.loc[:, 'xgb'].values.argsort()[-30:], :].to_csv('features/top_features_xgb.csv')

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
    

    # ######################### train on Logistic ###################################
    # features = [
    #     'cons_tot_m12_visits',
    #     'cf_prob_max',
    #     'als_m6_id_rel_allnum',
    #     'als_m12_id_nbank_tot_mons',
    #     'ir_m3_id_x_cell_cnt',
    #     'ir_m6_id_x_name_cnt'
    # ]
    # pop_features = [

    # ]
    # ds_final = 'data/merge_selected'
    # ds_t = pd.read_csv(ds_path, encoding='gb18030', header=0, index_col=0)
    # pop_t = pd.read_csv('data/pop.csv', encoding='gb18030', header=0, index_col=0)
    # pd.concat([pop_t.loc[:, pop_features], ds_t.loc[:, features], ds_t.iloc[:, -1]], axis=1).to_csv(ds_final)
    # ds_t = pd.read_csv(ds_final, header=0, index_col=0)
    # Preprocess.fill_cat(ds_final, ds_t.columns[:-1], save_path=ds_final)
    # ds_t = pd.read_csv(ds_final, header=0, index_col=0)
    # Preprocess.fill_na(ds_final, ds_t.columns[:-1], replacement=-1, save_path=ds_final)
    # ds_t = pd.read_csv(ds_final, header=0, index_col=0)
    # xgb_t = XGBClassifier()
    # # for column in ds_t.columns[:-1]:
    # #     printlog(column)
    # #     printlog(len(Temp_support.feature_woe(ds_final, column, -1)[0]))
    # #     ds_t.loc[:, column] = Temp_support.feature_woe(ds_final, column, -1)[0]
    # # ds_t.to_csv('data/merge_selected_woe.csv')

    # # ds_t = pd.read_csv('data/merge_selected_woe.csv', header=0, index_col=0)
    # # xgb_t.fit(train_fe, train_lb)
    # # prediction = xgb_t.predict_proba(test_fe).tolist()
    # train_fe, test_fe, train_lb, test_lb = train_test_split(ds_t.iloc[:, :-1], ds_t.iloc[:, -1], train_size=0.7, random_state=1)
    # # for i, pre in enumerate(prediction):
    # #     prediction[i] = pre[1]
    # # plt.scatter(prediction, test_lb, s=0.3, label='测试集表现')
    # # plt.title('XGB预测表现')
    # # plt.xlabel('XGB预测值分布')
    # # plt.ylabel('测试集标签值分布')
    # # plt.legend()
    # # plt.savefig('misc/xgb_1.png')
    # # plt.close()
    # # sns.distplot(prediction, bins=15, label='XGB预测值')
    # # sns.distplot(test_lb,    bins=15, label='测试集标签值')
    # # plt.title('XGB预测表现KDE-直方图')
    # # plt.xlabel('标签/预测值')
    # # plt.ylabel('标签/预测值分布')
    # # plt.legend()
    # # plt.savefig('misc/xgb.png')
    # # plt.close()
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


























    # #++++++++++++++++++++++++ data cleaning ++++++++++++++++++++++++
# abstract special features
    # ##################### special features(like dtype='O') #####################
    # special_features = []
    # for feature_class, flag in zip(classed_features, flag_list):
    #     special_features.extend(Preprocess.special_feature(ds_smp_path, feature_class, encoding='gb18030'))
    # Preprocess.clean_feature(ds_smp_path, special_features, save_path=ds_smp_path, encoding='gb18030')
    # classed_features = Preprocess.pattern_to_feature(ds_smp_path, check_feature_pattern)
# check if outlier data are of same label
    # ##################### outlier data #####################
    # ## if outlier is set na, then is treated as missing data in following process
    # for feature_class, flag in zip(classed_features, flag_list):
    #     Preprocess.clean_outlier(ds_smp_path, feature_class, threshold=1, encoding='gb18030', save_path=ds_smp_path)
    #     EDA.feature_EDA(ds_smp_path, feature_class[:20], encoding='gb18030')
    #     EDA.feature_na(ds_smp_path, feature_class[:20], encoding='gb18030')
    
    # ##################### poor sample #####################
    # EDA_massive.poor_sample(ds_smp_path, 9, encoding='gb18030')
    # Preprocess.clean_poor_sample(ds_smp_path, 9, save_path=ds_smp_path, encoding='gb18030')
    # EDA_massive.poor_sample(ds_smp_path, 9, encoding='gb18030')

    # ##################### poor feature #####################
    # EDA_massive.poor_feature(ds_smp_path, 3, encoding='gb18030')
    # Preprocess.clean_poor_feature(ds_smp_path, 3, save_path=ds_smp_path, encoding='gb18030')
    # EDA_massive.poor_feature(ds_smp_path, 3, encoding='gb18030')
    # classed_features = Preprocess.pattern_to_feature(ds_smp_path, check_feature_pattern)

# to be modified
    # ##################### dull feature #####################
    # EDA_massive.dull_feature(ds_smp_path, 0.9, -1, encoding='gb18030')
    # Preprocess.clean_dull_feature(ds_smp_path, 0.9, -1, save_path=ds_smp_path, encoding='gb18030')
    # EDA_massive.dull_feature(ds_smp_path, 0.9, -1, encoding='gb18030')
    # classed_features = Preprocess.pattern_to_feature(ds_smp_path, check_feature_pattern)

    # ##################### missing data #####################
    # EDA.feature_EDA(ds_smp_path, flag_list, encoding='gb18030')
    # for feature_class, flag in zip(classed_features, flag_list):
    #     Preprocess.fill_na(ds_smp_path, feature_class, flag_feature=flag, flag_replacement=-1, save_path=ds_smp_path, encoding='gb18030')
    #     EDA.feature_na(ds_smp_path, feature_class[:20], encoding='gb18030')

# todo
    # ##################### information value #####################
    # Preprocess.clean_lowIV_feature(ds_smp_path, -1, encoding='gb18030')

    # ##################### duplicated sample #####################
    # protect the label column when cleaning should be noticed

# done
    # ##################### sort by user_date #####################
    # EDA_massive.date_feature(ds_smp_path, feature='user_date', labels=labels, label_column=-1, file_path='tmp/record_user_date_count.png')
    # Preprocess.sort(ds_smp_path, ds_smp_srt_path, 'user_date', encoding='gb18030')
    # EDA_massive.EDA(ds_smp_srt_path, 'feature', encoding='gb18030')

    # ##################### split train/test set #####################
    # ## separate train & test datasets from sorted dataset
    # feature_train, label_train, feature_test, label_test = Preprocess.split_train_test_set(ds_smp_srt_path, sample_datasets, train_rate=0.7, shuffle=False, encoding='gb18030')
    # EDA_massive.EDA(sample_datasets[0], 'feature', encoding='gb18030')
    # EDA_massive.EDA(sample_datasets[1], 'label', encoding='gb18030')
    # EDA_massive.EDA(sample_datasets[2], 'feature', encoding='gb18030')
    # EDA_massive.EDA(sample_datasets[3], 'label', encoding='gb18030')
    # Preprocess.split_measure(label_train, label_test, labels)
    

if __name__ == '__main__':
    run()
    winsound.Beep(600,1000)