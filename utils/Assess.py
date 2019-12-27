import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV, ParameterGrid
from sklearn.metrics._classification import confusion_matrix
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
import os
import matplotlib.colorbar as cb
from parfit import bestFit, fitModels, scoreModels, plotScores, getBestModel, getBestScore

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
valid_features, valid_label, grid_params, grid_scorers, refit_scorer, n_jobs=-1):
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
    printlog('Assess.gridCVSelection: {} started.'.format(estimator_name))
    grid = GridSearchCV(estimator, grid_params, grid_scorers, refit=refit_scorer, n_jobs=n_jobs)
    grid.fit(X=train_features, y=train_label)
    train_CV_result = grid.cv_results_
    grid.fit(X=valid_features, y=valid_label)
    valid_CV_result = grid.cv_results_
    if estimator_name and save_folder: 
        _plotGridCVResult(estimator_name, save_folder, grid_params, grid_scorers, train_CV_result, valid_CV_result)
    bias               = train_CV_result['mean_test_{}'.format(refit_scorer)] - valid_CV_result['mean_test_{}'.format(refit_scorer)]
    variance           = np.power(valid_CV_result['std_test_{}'.format(refit_scorer)], 2)
    error              = bias + variance
    expected_CV_result = train_CV_result['mean_test_{}'.format(refit_scorer)] + error
    printlog('Assess.gridCVSelection: optimal params: {}'.format(train_CV_result['params'][np.argmax(expected_CV_result)]))
    printlog('Assess.gridCVSelection: {} finished.'.format(estimator_name))
    return train_CV_result['params'][np.argmax(expected_CV_result)]


def _plotGridCVResult(estimator_name, save_folder, grid_params, grid_scorers, train_CV_result, valid_CV_result):
    assert len(list(grid_params.keys())) <= 2, 'Assess.gridCVSelection: when plotting, only grid_params less than two is acceptable.'
    # print('buttom_scorer_split_indices: {}'.format(bottom_scorer_step_indices))
    if len(list(grid_params.keys())) == 1:
        grid_params = {'default': [0], list(grid_params.keys())[0]: list(grid_params.values())[0]}
    plot_names      = []
    plot_indices    = list(grid_params.values())[1]
    bottom_scorer_step_indices = len(list(grid_params.values())[1])
    for scorer in grid_scorers:
        for param_value in list(grid_params.values())[0]:
            plot_names.append(os.path.join(save_folder, 'grid_{}_{}_{}_{}.png'.format(estimator_name, scorer, list(grid_params.keys())[0], param_value)))
    plot_names_iterator = 0
    for scorer in grid_scorers:
        ## train
        train_scorer_mean   = train_CV_result['mean_test_{}'.format(scorer)]
        train_scorer_var    = np.power(train_CV_result['std_test_{}'.format(scorer)], 2)
        # train_scorer_var    = train_CV_result['std_test_{}'.format(scorer)] # actually std
        train_scorer_lower_var = train_scorer_mean - train_scorer_var / 2
        train_scorer_upper_var = train_scorer_mean + train_scorer_var / 2
        ## valid
        valid_scorer_mean   = valid_CV_result['mean_test_{}'.format(scorer)]
        valid_scorer_var    = np.power(valid_CV_result['std_test_{}'.format(scorer)], 2)
        # valid_scorer_var    = valid_CV_result['std_test_{}'.format(scorer)] # actually std
        valid_scorer_lower_var = valid_scorer_mean - valid_scorer_var / 2
        valid_scorer_upper_var = valid_scorer_mean + valid_scorer_var / 2
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


def confusionMatrixFromPrediction(y_true, y_pred, display_labels, title, normalize='true', cmap=plt.cm.Blues, save_path=None):
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=display_labels)
    plot = disp.plot(cmap=cmap, xticks_rotation='horizontal')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    return plot


def gridTrainValidSelection(estimator, grid, X_train, y_train, X_valid, y_valid, metric=roc_auc_score, scoreLabel='ROC AUC', to_file=None, 
showPlot=True, n_jobs=-1, verbose=10, predict_proba=True, greater_is_better=True, vrange=None, cmap=plt.cm.Blues):
    paramGrid = ParameterGrid(grid)
    printlog("-------------FITTING MODELS-------------")
    models = fitModels(estimator, paramGrid, X_train, y_train, n_jobs, verbose)
    printlog("-------------SCORING MODELS-------------")
    scores = scoreModels(models, X_valid, y_valid, metric, predict_proba, n_jobs, verbose)
    if showPlot:
        _plotScores(scores, paramGrid, to_file, scoreLabel, greater_is_better, vrange, cmap)

    return getBestModel(models, scores, greater_is_better), getBestScore(scores, greater_is_better), models, scores


def _plotScores(scores, paramGrid, to_file, scoreLabel=None, greater_is_better=True, vrange=None, cmap="YlOrRd"):
    keys = sorted(list(paramGrid)[0].keys())
    uniqParams = dict()
    order = dict()
    for k in keys:
        order[k] = np.unique([str(params[k]) for params in list(paramGrid)], return_index=True)[1]
        uniqParams[k] = [params[k] for params in np.asarray(list(paramGrid))[sorted(order[k])]]

    keysToPlot = list()
    for k in keys:
        if len(uniqParams[k]) > 1:
            keysToPlot.append(k)

    for k in keys:
        if k not in keysToPlot:
            uniqParams.pop(k, None)

    numDim = len(keysToPlot)
    if numDim > 3:
        printlog("Too many dimensions to plot.")
    elif numDim == 3:
        _plot3DGrid(scores, uniqParams, keysToPlot, scoreLabel, greater_is_better, vrange, cmap, to_file)
    elif numDim == 2:
        _plot2DGrid(scores, uniqParams, keysToPlot, scoreLabel, greater_is_better, vrange, cmap, to_file)
    elif numDim == 1:
        _plot1DGrid(scores, uniqParams, scoreLabel, vrange, to_file)
    else:
        printlog("No parameters that vary in the grid")

def _plot3DGrid(scores, paramsToPlot, keysToPlot, scoreLabel, greater_is_better, vrange, cmap, to_file):
    vmin = np.min(scores)
    vmax = np.max(scores)
    scoreGrid = np.reshape(
        scores, (len(paramsToPlot[keysToPlot[0]]), len(paramsToPlot[keysToPlot[1]]), len(paramsToPlot[keysToPlot[2]]))
    )

    smallest_dim = np.argmin(scoreGrid.shape)
    if smallest_dim != 2:
        scoreGrid = np.swapaxes(scoreGrid, smallest_dim, 2)
        keysToPlot[smallest_dim], keysToPlot[2] = keysToPlot[2], keysToPlot[smallest_dim]

    nelements = scoreGrid.shape[2]
    nrows = np.floor(nelements ** 0.5).astype(int)
    ncols = np.ceil(1.0 * nelements / nrows).astype(int)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex="all",
        sharey="all",
        figsize=(
            int(round(len(paramsToPlot[keysToPlot[1]]) * ncols * 1.33)),
            int(round(len(paramsToPlot[keysToPlot[0]]) * nrows * 1.33)),
        ),
    )

    if not greater_is_better:
        if cmap.endswith("_r"):
            cmap = cmap[:-2]
        else:
            cmap = cmap + "_r"
    i = 0
    for ax in axes.flat:
        if vrange is not None:
            im = ax.imshow(scoreGrid[:, :, i], cmap=cmap, vmin=vrange[0], vmax=vrange[1])
        else:
            im = ax.imshow(scoreGrid[:, :, i], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel(keysToPlot[1])
        ax.set_xticks(np.arange(len(paramsToPlot[keysToPlot[1]])))
        ax.set_xticklabels(paramsToPlot[keysToPlot[1]])
        ax.set_ylabel(keysToPlot[0])
        ax.set_yticks(np.arange(len(paramsToPlot[keysToPlot[0]])))
        ax.set_yticklabels(paramsToPlot[keysToPlot[0]])
        ax.set_title(keysToPlot[2] + " = " + str(paramsToPlot[keysToPlot[2]][i]))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        i += 1
        if i == nelements:
            break
    if scoreLabel is not None:
        fig.suptitle(scoreLabel, fontsize=18)
    else:
        fig.suptitle("Score", fontsize=18)
    fig.subplots_adjust(right=0.8)
    cbar = cb.make_axes(ax, location="right", fraction=0.03)
    fig.colorbar(im, cax=cbar[0])
    plt.savefig(to_file)
    plt.close()


def _plot2DGrid(scores, paramsToPlot, keysToPlot, scoreLabel, greater_is_better, vrange, cmap, to_file):
    scoreGrid = np.reshape(scores, (len(paramsToPlot[keysToPlot[0]]), len(paramsToPlot[keysToPlot[1]])))
    plt.figure(
        figsize=(
            int(round(len(paramsToPlot[keysToPlot[1]]) / 1.33)),
            int(round(len(paramsToPlot[keysToPlot[0]]) / 1.33)),
        )
    )
    if not greater_is_better:
        if cmap.endswith("_r"):
            cmap = cmap[:-2]
        else:
            cmap = cmap + "_r"
    if vrange is not None:
        plt.imshow(scoreGrid, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
    else:
        plt.imshow(scoreGrid, cmap=cmap)
    plt.xlabel(keysToPlot[1])
    plt.xticks(np.arange(len(paramsToPlot[keysToPlot[1]])), paramsToPlot[keysToPlot[1]])
    plt.ylabel(keysToPlot[0])
    plt.yticks(np.arange(len(paramsToPlot[keysToPlot[0]])), paramsToPlot[keysToPlot[0]])
    if scoreLabel is not None:
        plt.title(scoreLabel)
    else:
        plt.title("Score")
    plt.colorbar()
    plt.box(on=False)
    plt.savefig(to_file)
    plt.close()


def _plot1DGrid(scores, paramsToPlot, scoreLabel, vrange, to_file):
    key = list(paramsToPlot.keys())
    plt.figure(figsize=(int(round(len(paramsToPlot[key[0]]) / 1.33)), 6))
    plt.plot(np.linspace(0, len(paramsToPlot[key[0]]), len(scores)), scores, "-or")
    plt.xlabel(key[0])
    plt.xticks(np.linspace(0, len(paramsToPlot[key[0]]), len(scores)), paramsToPlot[key[0]])
    if scoreLabel is not None:
        plt.ylabel(scoreLabel)
    else:
        plt.ylabel("Score")
    if vrange is not None:
        plt.ylim(vrange[0], vrange[1])
    plt.box(on=False)
    plt.savefig(to_file)
    plt.close()