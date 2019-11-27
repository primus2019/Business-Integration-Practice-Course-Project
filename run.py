from utils import EDA_massive, Preprocess, Log, EDA, Feature_selection, Model, Temp_support

import numpy as np
import pandas as pd
from functools import reduce
import itertools
from sklearn import tree
from utils.Log import printlog
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

    Log.clear_log(creative=True)
    Log.clear_log(file_path=feature_selection_log, creative=True)
    # ##################### EDA #####################
    # EDA_massive.EDA(ds_path, 'feature', encoding='gb18030')
    # EDA_massive.date_feature(ds_path, 'user_date', [0, 1], -1, 'tmp/date_feature.png', encoding='gb18030')
    #################### split sub dataset for test modelling #####################
    # Preprocess.split(ds_path, ds_smp_path, chunksize=1000, encoding='gb18030')
    # Preprocess.split(ds_path, ds_smp_path, fraction=1, shuffle=False, encoding='gb18030') ###
    # EDA_massive.EDA(ds_smp_path, 'feature', folder='tmp', encoding='gb18030')
    # EDA_massive.date_feature(ds_smp_path, 'user_date', [0, 1], -1, 'tmp/record_user_date_count.png', encoding='gb18030')
    ##################### necessary for afterward debugging #####################
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
    # ##################### class 3 - ir #####################
    # printlog('class 3 - ir')
    # ## class 3 variables
    # ds_c3       = 'tmp/ds_c3_ir.csv'
    # ds_c3_na    = 'tmp/ds_c3_ir_na.csv'
    # ds_c3_cut_1 = 'tmp/ds_c3_ir_cut_1.csv'
    # ds_c3_cut_2 = 'tmp/ds_c3_ir_cut_2.csv'
    # ds_c3_iv_cut_1 = 'iv/ds_c3_ir_iv_cut_1.csv'
    # ds_c3_iv_cut_2 = 'iv/ds_c3_ir_iv_cut_2.csv'
    # fe_c3_pattern = '^ir_'
    # fe_c3       = Preprocess.pattern_to_feature(ds_path, fe_c3_pattern, encoding='gb18030')[0]
    # log_fe_c3_iv_1  = 'features/fe_c3_ir_iv_1.log'
    # log_fe_c3_iv_2  = 'features/fe_c3_ir_iv_2.log'
    # ds_t = pd.read_csv(ds_path, encoding='gb18030', header=0, index_col=0)
    # pd.concat([ds_t.loc[:, fe_c3], ds_t.iloc[:, -1]], axis=1).to_csv(ds_c3, encoding='gb18030')
    # ## start of selection
    # printlog('class 3 - fill na')
    # Preprocess.fill_na(ds_c3, fe_c3, replacement=-1, save_path=ds_c3_na, encoding='gb18030')
    # printlog('class 3 - cut 1')
    # Temp_support.cut(ds_c3_na, fe_c3, threshold=5, bin=5,   save_path=ds_c3_cut_1, encoding='gb18030')
    # printlog('class 3 - cut 2')
    # Temp_support.cut(ds_c3_na, fe_c3, threshold=10, bin=10, save_path=ds_c3_cut_2, encoding='gb18030')
    # printlog('class 3 - select by iv 1')
    # fe_c3_iv_1 = Temp_support.select_feature_iv(ds_c3_cut_1, fe_c3, -1, 0.5, 0.3, to_file=ds_c3_iv_cut_1, encoding='gb18030')
    # printlog('class 3 - select by iv 2')
    # fe_c3_iv_2 = Temp_support.select_feature_iv(ds_c3_cut_2, fe_c3, -1, 0.5, 0.3, to_file=ds_c3_iv_cut_2, encoding='gb18030')
    # printlog('class 3 - saving')
    # Log.itersave(file_path=log_fe_c3_iv_1, iteritem=fe_c3_iv_1)
    # Log.itersave(file_path=log_fe_c3_iv_2, iteritem=fe_c3_iv_2)
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
    printlog('class 5 - als')
    ## class 5 variables
    ds_c5                 = 'tmp/ds_c5_als.csv'
    ds_c5_varied          = 'tmp/ds_c5_als_varied.csv'
    ds_c5_varied_na       = 'tmp/ds_c5_als_varied_na.csv'
    ds_c5_varied_cut_1 = 'tmp/ds_c5_als_varied_cut_1.csv'
    ds_c5_varied_cut_2 = 'tmp/ds_c5_als_varied_cut_2.csv'
    ds_c5_varied_cut_1_iv = 'iv/ds_c5_als_varied_iv_cut_1.csv'
    ds_c5_varied_cut_2_iv = 'iv/ds_c5_als_varied_iv_cut_2.csv'
    fe_c5_pattern   = '^als_'
    log_fe_c5_iv_1  = 'features/fe_c5_als_iv_1.log'
    log_fe_c5_iv_2  = 'features/fe_c5_als_iv_2.log'
    als_preffix = [
        ['^als_d7_id_', '^als_d15_id_', '^als_m1_id_', '^als_m3_id_', '^als_m6_id_', '^als_m12_id_', '^als_fst_id_', '^als_lst_id_'],
        ['^als_d7_cell_', '^als_d15_cell_', '^als_m1_cell_', '^als_m3_cell_', '^als_m6_cell_', '^als_m12_cell_', '^als_fst_cell_', '^als_lst_cell_']]
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
    # Log.itersave(file_path=log_fe_c5_iv_1, iteritem=fe_c5_iv_1)
    # Log.itersave(file_path=log_fe_c5_iv_2, iteritem=fe_c5_iv_2)
    # ##################### class 6 - cf #####################
    # printlog('class 6 - cf')
    # ## class 6 variables
    # ds_c6       = 'tmp/ds_c6_ir.csv'
    # ds_c6_na    = 'tmp/ds_c6_ir_na.csv'
    # ds_c6_cut_1 = 'tmp/ds_c6_ir_cut_1.csv'
    # ds_c6_cut_2 = 'tmp/ds_c6_ir_cut_2.csv'
    # ds_c6_iv_cut_1 = 'iv/ds_c6_ir_iv_cut_1.csv'
    # ds_c6_iv_cut_2 = 'iv/ds_c6_ir_iv_cut_2.csv'
    # fe_c6_pattern = '^ir_'
    # fe_c6       = Preprocess.pattern_to_feature(ds_path, fe_c6_pattern, encoding='gb18030')[0]
    # log_fe_c6_iv_1  = 'features/fe_c6_ir_iv_1.log'
    # log_fe_c6_iv_2  = 'features/fe_c6_ir_iv_2.log'
    # ds_t = pd.read_csv(ds_path, encoding='gb18030', header=0, index_col=0)
    # pd.concat([ds_t.loc[:, fe_c6], ds_t.iloc[:, -1]], axis=1).to_csv(ds_c6, encoding='gb18030')
    # ## start of selection
    # printlog('class 6 - fill na')
    # Preprocess.fill_na(ds_c6, fe_c6, replacement=-1, save_path=ds_c6_na, encoding='gb18030')
    # printlog('class 6 - cut 1')
    # Temp_support.cut(ds_c6_na, fe_c6, threshold=5, bin=5,   save_path=ds_c6_cut_1, encoding='gb18030')
    # printlog('class 6 - cut 2')
    # Temp_support.cut(ds_c6_na, fe_c6, threshold=10, bin=10, save_path=ds_c6_cut_2, encoding='gb18030')
    # printlog('class 6 - select by iv 1')
    # fe_c6_iv_1 = Temp_support.select_feature_iv(ds_c6_cut_1, fe_c6, -1, 0.5, 0.3, to_file=ds_c6_iv_cut_1, encoding='gb18030')
    # printlog('class 6 - select by iv 2')
    # fe_c6_iv_2 = Temp_support.select_feature_iv(ds_c6_cut_2, fe_c6, -1, 0.5, 0.3, to_file=ds_c6_iv_cut_2, encoding='gb18030')
    # printlog('class 6 - saving')
    # Log.itersave(file_path=log_fe_c6_iv_1, iteritem=fe_c6_iv_1)
    # Log.itersave(file_path=log_fe_c6_iv_2, iteritem=fe_c6_iv_2)
    # ##################### class 7 - cons #####################
    # printlog('class 7 - cf')
    # ## class 7 variables
    # ds_c7       = 'tmp/ds_c7_ir.csv'
    # ds_c7_na    = 'tmp/ds_c7_ir_na.csv'
    # ds_c7_cut_1 = 'tmp/ds_c7_ir_cut_1.csv'
    # ds_c7_cut_2 = 'tmp/ds_c7_ir_cut_2.csv'
    # ds_c7_iv_cut_1 = 'iv/ds_c7_ir_iv_cut_1.csv'
    # ds_c7_iv_cut_2 = 'iv/ds_c7_ir_iv_cut_2.csv'
    # fe_c7_pattern = '^ir_'
    # fe_c7       = Preprocess.pattern_to_feature(ds_path, fe_c7_pattern, encoding='gb18030')[0]
    # log_fe_c7_iv_1  = 'features/fe_c7_ir_iv_1.log'
    # log_fe_c7_iv_2  = 'features/fe_c7_ir_iv_2.log'
    # ds_t = pd.read_csv(ds_path, encoding='gb18030', header=0, index_col=0)
    # pd.concat([ds_t.loc[:, fe_c7], ds_t.iloc[:, -1]], axis=1).to_csv(ds_c7, encoding='gb18030')
    # ## start of selection
    # printlog('class 7 - fill na')
    # Preprocess.fill_na(ds_c7, fe_c7, replacement=-1, save_path=ds_c7_na, encoding='gb18030')
    # printlog('class 7 - cut 1')
    # Temp_support.cut(ds_c7_na, fe_c7, threshold=5, bin=5,   save_path=ds_c7_cut_1, encoding='gb18030')
    # printlog('class 7 - cut 2')
    # Temp_support.cut(ds_c7_na, fe_c7, threshold=10, bin=10, save_path=ds_c7_cut_2, encoding='gb18030')
    # printlog('class 7 - select by iv 1')
    # fe_c7_iv_1 = Temp_support.select_feature_iv(ds_c7_cut_1, fe_c7, -1, 0.5, 0.3, to_file=ds_c7_iv_cut_1, encoding='gb18030')
    # printlog('class 7 - select by iv 2')
    # fe_c7_iv_2 = Temp_support.select_feature_iv(ds_c7_cut_2, fe_c7, -1, 0.5, 0.3, to_file=ds_c7_iv_cut_2, encoding='gb18030')
    # printlog('class 7 - saving')
    # Log.itersave(file_path=log_fe_c7_iv_1, iteritem=fe_c7_iv_1)
    # Log.itersave(file_path=log_fe_c7_iv_2, iteritem=fe_c7_iv_2)

    
    # ##################### visualization class 5 - als #####################
    ds_t = pd.read_csv(ds_c5, encoding='gb18030', header=0, index_col=0)
    for i, prefix_class in enumerate(als_preffix):
        df_list = []
        for p in prefix_class:
            features = Preprocess.pattern_to_feature(ds_t, p, encoding='gb18030')[0]
            printlog('doing {}'.format(p))
            for feature in features:
                count_all   = ds_t.shape[0]
                count_na    = ds_t.loc[:, feature].isna().sum()
                count_notna = ds_t.loc[:, feature].notna().sum()
                count_value = len(list(set(np.ravel(ds_t.loc[ds_t[feature].notna(), feature].values))))
                df_list.append(pd.DataFrame([count_all, count_na, count_notna, count_value], 
                    columns=[feature], index=['count_all', 'count_na', 'count_notna', 'count_value']))
                # printlog(df_list[-1])
        pd.concat(df_list, axis=1).to_csv('misc/als_{}.csv'.format(i))
            

    winsound.Beep(600,1000)



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