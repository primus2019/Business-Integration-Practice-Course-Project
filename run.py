from utils import EDA_massive, Preprocess, Log, EDA


def run():
    ds_path = 'data/data.csv'
    ds_smp_path = 'tmp/ds_smp.csv'
    flag_list = ['flag_specialList_c', 'flag_fraudrelation_g', 'flag_inforelation', 'flag_applyloanusury', 'flag_applyloanstr', 'flag_ConsumptionFeature', 'flag_consumption_c']
    check_feature_pattern = ['^sl_', '^frg_', '^ir_', '^alu_', '^als_', '^cf_', '^cons_']
    ds_smp_srt_path = 'tmp/ds_smp_srt.csv'
    sample_datasets = [
        'tmp/ds_feature_train.csv',
        'tmp/ds_label_train.csv',
        'tmp/ds_feature_test.csv',
        'tmp/ds_label_test.csv'
    ]

    Log.clear_log(creative=True)
    # # originla dataset path
    # # EDA(checking na data, feature type)
    # EDA_massive.EDA(ds_path, 'feature', encoding='gb18030')
    # ## sample dataset path(a smaller one)
    # ## split sub dataset for test modelling
    # Preprocess.split(ds_path, ds_smp_path, chunksize=1000, encoding='gb18030')
    # EDA_massive.EDA(ds_smp_path, 'feature', folder='tmp', save_graph=True, encoding='gb18030')
    classed_features = Preprocess.pattern_to_feature(ds_smp_path, check_feature_pattern)


    # # preprocess
    # ## data cleaning
    # ##################### special features(like dtype='O') #####################
    # special_features = []
    # for feature_class, flag in zip(classed_features, flag_list):
    #     special_features.extend(Preprocess.special_feature(ds_smp_path, feature_class, encoding='gb18030'))
    # Preprocess.clean_special_feature(ds_smp_path, special_features, save_path=ds_smp_path, encoding='gb18030')
    classed_features = Preprocess.pattern_to_feature(ds_smp_path, check_feature_pattern)
    # for i, feature_class in enumerate(classed_features):
    #     for special_feature in special_features:
    #         if special_feature in feature_class:
    #             classed_features[i].remove(special_feature)
    #             print(1)
    # ##################### outlier data(if outlier is set na, then is treated as missing data in following process) #####################
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
    classed_features = Preprocess.pattern_to_feature(ds_smp_path, check_feature_pattern)
    # ##################### missing data #####################
    # EDA.feature_EDA(ds_smp_path, flag_list, encoding='gb18030')
    # for feature_class, flag in zip(classed_features, flag_list):
    #     Preprocess.fill_na(ds_smp_path, feature_class, flag_feature=flag, flag_replacement=-1, save_path=ds_smp_path, encoding='gb18030')
    #     EDA.feature_na(ds_smp_path, feature_class[:20], encoding='gb18030')
    

# todo

    # ##################### duplicated sample #####################
    
    # ## sort sample dataset by 'user_date' in ascend
    # labels = EDA_massive.labels(ds_smp_srt_path, column=-1, encoding='gb18030')
    # print(labels)
    # EDA_massive.date_feature(ds_smp_path, feature='user_date', labels=labels, label_column=-1, file_path='tmp/record_user_date_count.png')
    # Preprocess.sort(ds_smp_path, ds_smp_srt_path, 'user_date', encoding='gb18030')
    # EDA_massive.EDA(ds_smp_srt_path, 'feature', encoding='gb18030')
    # ## separate train & test datasets from sorted dataset
    # feature_train, label_train, feature_test, label_test = Preprocess.split_train_test_set(ds_smp_srt_path, sample_datasets, train_rate=0.7, shuffle=False, encoding='gb18030')
    # EDA_massive.EDA(sample_datasets[0], 'feature', encoding='gb18030')
    # EDA_massive.EDA(sample_datasets[1], 'label', encoding='gb18030')
    # EDA_massive.EDA(sample_datasets[2], 'feature', encoding='gb18030')
    # EDA_massive.EDA(sample_datasets[3], 'label', encoding='gb18030')
    # Preprocess.split_measure(label_train, label_test, labels)
    


    






if __name__ == '__main__':
    run()