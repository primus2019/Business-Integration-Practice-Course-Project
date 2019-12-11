import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
import seaborn as sns
from sklearn.model_selection import cross_val_score
import os


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


def xgbAssess(y_true, y_pred, save_folder):
    plt.scatter(y_pred[:, 1].ravel(), y_true.ravel(), s=0.3, label='测试集表现')
    plt.title('XGB预测表现 - 散点分布')
    plt.xlabel('XGB预测值分布')
    plt.ylabel('测试集标签值分布')
    plt.legend()
    plt.savefig(os.path.join(save_folder, 'xgb_scatter.png'))
    plt.close()
    sns.distplot(y_pred[:, 1].ravel(), bins=15, label='XGB预测值')
    sns.distplot(y_true.ravel(),       bins=15, label='测试集标签值')
    plt.title('XGB预测表现 - KDE直方图')
    plt.xlabel('标签/预测值')
    plt.ylabel('标签/预测值分布')
    plt.legend()
    plt.savefig(os.path.join(save_folder, 'xgb_bar.png'))
    plt.close()

    KSplot(y_true, y_pred, os.path.join(save_folder, 'xgb_ks.png'), 'XGB预测表现 - KS图')
    ROCplot(y_true, y_pred, os.path.join(save_folder, 'xgb_roc.png'), 'XGB预测表现 - ROC图')


def estimatorAssess(estimator, X, y, scoring, cv):
    cross_val_score(estimator, X, y, scoring=scoring, cv=cv)
    