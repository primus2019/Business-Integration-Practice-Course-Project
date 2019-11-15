from utils import EDA_massive, Preprocess, Log


def run():
    # Log.clear_log(creative=True)
    # ## originla dataset path
    # ds_path = 'data/data.csv'
    # ## EDA(checking na data, feature type)
    # EDA_massive.EDA(ds_path, 'feature', encoding='gb18030')
    # ## sample dataset path(a smaller one)
    # ds_smp_path = 'tmp/ds_smp.csv'
    # Preprocess.split(ds_path, ds_smp_path, chunksize=1000, encoding='gb18030')
    # EDA_massive.EDA(ds_smp_path, 'feature', folder='tmp', save_graph=True, encoding='gb18030')
    # ## sort sample dataset by 'user_date' in ascend
    ds_smp_srt_path = 'tmp/ds_smp_srt.csv'
    # EDA_massive.date_feature(ds_smp_path, 'user_date', 'tmp/record_user_date_count.png')
    # Preprocess.sort(ds_smp_path, ds_smp_srt_path, 'user_date', encoding='gb18030')
    # EDA_massive.EDA(ds_smp_srt_path, 'feature', encoding='gb18030')
    # ## separate train & test datasets from sorted dataset
    sample_datasets = [
        'tmp/ds_feature_train.csv',
        'tmp/ds_label_train.csv',
        'tmp/ds_feature_test.csv',
        'tmp/ds_label_test.csv'
    ]
    feature_train, label_train, feature_test, label_test = Preprocess.split_train_test_set(ds_smp_srt_path, sample_datasets, train_rate=0.7, shuffle=False, encoding='gb18030')
    # EDA_massive.EDA(sample_datasets[0], 'feature', encoding='gb18030')
    # EDA_massive.EDA(sample_datasets[1], 'label', encoding='gb18030')
    # EDA_massive.EDA(sample_datasets[2], 'feature', encoding='gb18030')
    # EDA_massive.EDA(sample_datasets[3], 'label', encoding='gb18030')
    Preprocess.split_measure(label_train, label_test, EDA_massive.labels(label_test, encoding='gb18030'))






if __name__ == '__main__':
    run()