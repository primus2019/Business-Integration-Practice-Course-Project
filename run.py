from utils import EDA_massive, Preprocess, Log, EDA, Feature_selection, Model, Temp_support

import numpy as np
import pandas as pd
from functools import reduce
import itertools
from sklearn import tree
from utils.Log import printlog

def run():
    ds_path = 'data/data.csv'
    ds_smp_path = 'tmp/ds_smp.csv'
    # ds_smp_path = ds_path
    ds_smp_spe_path = 'tmp/ds_smp_spe.csv'
    ds_smp_srt_path = 'tmp/ds_smp_srt.csv'
    flag_list = ['flag_specialList_c', 'flag_fraudrelation_g', 'flag_inforelation', 'flag_applyloanusury', 'flag_applyloanstr', 'flag_ConsumptionFeature', 'flag_consumption_c']
    check_feature_pattern = ['^sl_', '^frg_', '^ir_', '^alu_', '^als_', '^cf_', '^cons_']
    sample_datasets = [
        'tmp/ds_feature_train.csv',
        'tmp/ds_label_train.csv',
        'tmp/ds_feature_test.csv',
        'tmp/ds_label_test.csv'
    ]

    Log.clear_log(creative=True)
    # ##################### EDA #####################
    # EDA_massive.EDA(ds_path, 'feature', encoding='gb18030')
    # EDA_massive.date_feature(ds_path, 'user_date', [0, 1], -1, 'tmp/date_feature.png', encoding='gb18030')
    #################### split sub dataset for test modelling #####################
    Preprocess.split(ds_path, ds_smp_path, chunksize=1000, encoding='gb18030')
    # EDA_massive.EDA(ds_smp_path, 'feature', folder='tmp', encoding='gb18030')
    # EDA_massive.date_feature(ds_smp_path, 'user_date', [0, 1], -1, 'tmp/record_user_date_count.png', encoding='gb18030')
    ##################### necessary for afterward debugging #####################
    classed_features = Preprocess.pattern_to_feature(ds_smp_path, check_feature_pattern)
    labels = EDA_massive.labels(ds_smp_srt_path, column=-1, encoding='gb18030')
    

    # #************************ preprocess ************************
    # ##################### data cleaning #####################
    # ## unnecessary; if data cleaning is performed, the proceeding fill_cat will not work
    # special_features = reduce(lambda accum, fc: np.concatenate([accum, Preprocess.special_feature(ds_smp_path, fc)]), classed_features, np.array([]))
    # Preprocess.pop_feature(ds_smp_path, special_features, save_path=ds_smp_path, pop_path=ds_smp_spe_path, encoding='gb18030')
    #################### feature selection #####################
    #################### class 1 - sl #####################
    class_1_preffix = ['^sl_id_', '^sl_cell_', '^sl_lm_cell_']
    mut_exc_1_feature = Temp_support.feature_padding(ds_smp_path, classed_features[0], class_1_preffix, encoding='gb18030')
    ## the class_1_gate_feature is essential for gateway classification
    class_1_gate_feature = Feature_selection.hit_positive_rate(ds_smp_path, mut_exc_1_feature, -1, 0.9, encoding='gb18030')
    tocheck_feature = Feature_selection.hit_positive_rate(ds_smp_path, mut_exc_1_feature, -1, 0.6, encoding='gb18030')
    tocheck_feature = [feature for feature in tocheck_feature if feature not in class_1_gate_feature]
    ## if the features are categorical features, put them into decision tree and derive the tree model
    fill_na = lambda x, f: Preprocess.fill_na(x, f, flag_feature=flag_list[0], save_path=ds_smp_path, flag_replacement=-1, encoding='gb18030')
    ## the class_1_tree is essential for tree classifier here
    if len(tocheck_feature) > 0:
        class_1_tree = Model.tree_classifier(
            ds=ds_smp_path, 
            features=tocheck_feature, 
            label_column=-1, 
            fill_na=fill_na, 
            encoding='gb18030', 
            export_path='tmp/class_1_tree.dot'
        )
    else:
        printlog('class 1 - SpecialList: no feature to check with tree')
    ##################### class 2 - fr #####################
    ## note that classed_features[1] is updated here for label binarizer, new ds is saved at save_path in fill_cat
    fill_na = lambda x, f: Preprocess.fill_na(x, f, flag_feature=flag_list[1], flag_replacement=-1, encoding='gb18030')
    fill_cat = lambda x, f: Preprocess.fill_cat(x, f, method='label_binarizer', save_path=ds_smp_path, encoding='gb18030')
    ## the class_2_tree, class_2_categorical_encoder, classed_feature[1] are essential for tree classifier here
    class_2_tree, class_2_categorical_encoder, classed_features[1] = Model.tree_classifier(
        ds=ds_smp_path, 
        features=classed_features[1], 
        label_column=-1, 
        fill_na=fill_na, 
        fill_cat=fill_cat, 
        export_path='tmp/class_2_tree.dot', 
        encoding='gb18030'
    )
    ##################### class 3 - ir #####################
    class_3_preffix = [] # add here
    mut_exc_3_feature = Temp_support.feature_padding(ds_smp_path, classed_features[2], class_3_preffix, encoding='gb18030')
    ## all features are put into model
    class_3_feature = Feature_selection.hit_rate(ds_smp_path, mut_exc_3_feature, threshold=0.05, encoding='gb18030')
    class_3_feature = Preprocess.fill_na(ds_smp_path, class_3_feature, flag_feature=flag_list[2], flag_replacement=-1, save_path=ds_smp_path, encoding='gb18030')
    ##################### class 4 - alu #####################
    class_4_preffix = [] # add here
    mut_exc_4_feature = Temp_support.feature_padding(ds_smp_path, classed_features[3], class_4_preffix, encoding='gb18030')
    ## the class_4_gate_feature is essential for gateway classification
    class_4_gate_feature = Feature_selection.hit_positive_rate(ds_smp_path, mut_exc_4_feature, -1, 0.9, encoding='gb18030')
    tocheck_feature = Feature_selection.hit_positive_rate(ds_smp_path, mut_exc_4_feature, -1, 0.6, encoding='gb18030')
    tocheck_feature = [feature for feature in tocheck_feature if feature not in class_4_gate_feature]
    ## if the features are categorical features, put them into decision tree and derive the tree model
    fill_na = lambda x, f: Preprocess.fill_na(x, f, flag_feature=flag_list[3], save_path=ds_smp_path, flag_replacement=-1, encoding='gb18030')
    ## the class_4_tree is essential for tree classifier here
    if len(tocheck_feature) > 0:
        class_4_tree = Model.tree_classifier(
            ds=ds_smp_path, 
            features=tocheck_feature, 
            label_column=-1, 
            fill_na=fill_na, 
            encoding='gb18030', 
            export_path='tmp/class_4_tree.dot'
        )
    else:
        printlog('class 4 - SpecialList: no feature to check with tree')
    ##################### class 5 - als #####################
    class_5_preffix = [] # add here
    mut_exc_5_feature = Temp_support.feature_padding(ds_smp_path, classed_features[4], class_5_preffix, encoding='gb18030')
    ## all features are put into model
    class_5_feature = Feature_selection.hit_rate(ds_smp_path, mut_exc_5_feature, threshold=0.05, encoding='gb18030')
    class_5_feature = Preprocess.fill_na(ds_smp_path, class_5_feature, flag_feature=flag_list[4], flag_replacement=-1, save_path=ds_smp_path, encoding='gb18030')
    ##################### class 6 - cf #####################
    class_6_preffix = [] # add here
    mut_exc_6_feature = Temp_support.feature_padding(ds_smp_path, classed_features[5], class_6_preffix, encoding='gb18030')
    ## all features are put into model
    class_6_feature = Feature_selection.hit_rate(ds_smp_path, mut_exc_6_feature, threshold=0.05, encoding='gb18030')
    class_6_feature = Preprocess.fill_na(ds_smp_path, class_6_feature, flag_feature=flag_list[5], flag_replacement=-1, save_path=ds_smp_path, encoding='gb18030')
    ##################### class 7 - cons #####################
    class_7_preffix = [] # add here
    mut_exc_7_feature = Temp_support.feature_padding(ds_smp_path, classed_features[6], class_7_preffix, encoding='gb18030')
    ## all features are put into model
    class_7_feature = Feature_selection.hit_rate(ds_smp_path, mut_exc_7_feature, threshold=0.05, encoding='gb18030')
    class_7_feature = Preprocess.fill_na(ds_smp_path, class_7_feature, flag_feature=flag_list[6], flag_replacement=-1, save_path=ds_smp_path, encoding='gb18030')

    



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