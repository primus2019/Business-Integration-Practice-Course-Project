import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV
import os

from utils.Log import printlog

def KSplot(y_true, y_pred, save_path, title):
    skplt.metrics.plot_ks_statistic(y_true, y_pred)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def ROCplot(y_true, y_pred, save_path, title):
    skplt.metrics.plot_roc(y_true, y_pred)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def modelAssess(y_true, y_pred, save_folder, model_name):
    plt.scatter(y_pred[:, 1].ravel(), y_true.ravel(), s=0.3, label='测试集表现')
    plt.title('{}预测表现 - 散点分布'.format(model_name))
    plt.xlabel('{}预测值分布'.format(model_name))
    plt.ylabel('测试集标签值分布')
    plt.legend()
    plt.savefig(os.path.join(save_folder, '{}_scatter.png'.format(model_name)))
    plt.close()
    sns.distplot(y_pred[:, 1].ravel(), bins=15, label='{}预测值'.format(model_name))
    sns.distplot(y_true.ravel(),       bins=15, label='测试集标签值')
    plt.title('{}预测表现 - KDE直方图'.format(model_name))
    plt.xlabel('标签/预测值')
    plt.ylabel('标签/预测值分布')
    plt.legend()
    plt.savefig(os.path.join(save_folder, '{}_bar.png'.format(model_name)))
    plt.close()

    KSplot(y_true, y_pred, os.path.join(save_folder, '{}_ks.png'.format(model_name)), '{}预测表现 - KS图'.format(model_name))
    ROCplot(y_true, y_pred, os.path.join(save_folder, '{}_roc.png'.format(model_name)), '{}预测表现 - ROC图'.format(model_name))


def estimatorAssess(estimator, X, y, scoring, cv):
    cross_val_score(estimator, X, y, scoring=scoring, cv=cv)


def gridCVSelection(estimator, estimator_name, save_folder, train_features, train_label, 
valid_features, valid_label, grid_params, grid_scorers, refit_scorer):
    """
    # Example: 
    ```
    xgb = XGBClassifier()
    train_dataset = pd.read_csv(ds_train, header=0, index_col=0)
    valid_dataset = pd.read_csv(ds_valid, header=0, index_col=0)
    xgb_params          = {'max_depth': [3, 4, 5], 'n_estimators': range(10, 301, 10)}
    xgb_scorer          = ['neg_mean_squared_error', 'roc_auc']
    Assess.gridCVSelection(xgb, 'xgb', 'misc', 
        train_dataset.loc[:, selected_features], train_dataset.iloc[:,-1], 
        valid_dataset.loc[:, selected_features], valid_dataset.iloc[:,-1], 
        xgb_params, xgb_scorer, refit_scorer='roc_auc')
    ```
    """
    printlog('Assess.gridCVSelection: started.')
    grid = GridSearchCV(estimator, grid_params, grid_scorers, refit=refit_scorer, n_jobs=-1)
    grid.fit(X=train_features, y=train_label)
    train_CV_result = grid.cv_results_
    grid.fit(X=valid_features, y=valid_label)
    valid_CV_result = grid.cv_results_
    plot_names      = ['grid_{}_{}_{}.png'.format(estimator_name, list(grid_params.keys())[0], value) for value in list(grid_params.values())[0]]
    plot_indices    = list(grid_params.values())[1]
    bottom_scorer_step_indices = len(list(grid_params.values())[1])
    # print('buttong_scorer_split_indices: {}'.format(bottom_scorer_step_indices))
    plot_names      = []
    for scorer in grid_scorers:
        for param_value in list(grid_params.values())[0]:
            plot_names.append(os.path.join(save_folder, 'grid_{}_{}_{}_{}.png'.format(estimator_name, scorer, list(grid_params.keys())[0], param_value)))
    plot_names_iterator = 0
    for scorer in grid_scorers:
        train_scorer_mean   = train_CV_result['mean_test_{}'.format(scorer)]
        train_scorer_var    = np.power(train_CV_result['std_test_{}'.format(scorer)], 2)
        train_scorer_lower_var = train_scorer_mean - train_scorer_var / 2
        train_scorer_upper_var = train_scorer_mean + train_scorer_var / 2
        valid_scorer_mean   = valid_CV_result['mean_test_{}'.format(scorer)]
        valid_scorer_std    = np.power(valid_CV_result['std_test_{}'.format(scorer)], 2)
        valid_scorer_lower_var = valid_scorer_mean - valid_scorer_std / 2
        valid_scorer_upper_var = valid_scorer_mean + valid_scorer_std / 2
        for step, _ in enumerate(list(grid_params.values())[0]):
            step_start = step       * bottom_scorer_step_indices
            step_end   = (step + 1) * bottom_scorer_step_indices
            # print('step_start: {}; step_end: {};'.format(step_start, step_end))
            plt.plot(plot_indices, train_scorer_mean[step_start: step_end])
            plt.plot(plot_indices, valid_scorer_mean[step_start: step_end])
            plt.fill_between(plot_indices, train_scorer_lower_var[step_start: step_end], train_scorer_upper_var[step_start: step_end], alpha=0.5)
            plt.fill_between(plot_indices, valid_scorer_lower_var[step_start: step_end], valid_scorer_upper_var[step_start: step_end], alpha=0.5)
            plt.title('{} Param Selection on GridSearchCV'.format(estimator_name))
            plt.xlabel(list(grid_params.keys())[1])
            plt.ylabel(scorer)
            plt.legend(['train score', 'validation score'])
            plt.savefig(plot_names[plot_names_iterator])
            plot_names_iterator += 1
            plt.close()
    bias = train_CV_result['mean_test_{}'.format(refit_scorer)] - valid_CV_result['mean_test_{}'.format(refit_scorer)]
    variance = np.power(valid_CV_result['std_test_{}'.format(refit_scorer)], 2)
    error = bias + variance
    printlog('Assess.gridCVSelection: optimal params: {}'.format(train_CV_result['params'][np.argmin(error)]))
    printlog('Assess.gridCVSelection: finished.')
    return train_CV_result['params'][np.argmin(error)]

