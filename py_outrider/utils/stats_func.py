import numpy as np
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import (roc_curve, roc_auc_score,
                             precision_recall_curve, auc)
import warnings


def multiple_testing_nan(X_pvalue, method='fdr_by'):
    """
    applies multiple testing correction while dealing with nan values
    :param X_pvalue: array of p-values
    :param method: name of correction method
        (from statsmodels.stats.multitest.multipletests)
    :return: array of corrected p-values with nan values
    """
    mask = np.isfinite(X_pvalue)
    pval_corrected = np.empty(X_pvalue.shape)
    pval_corrected.fill(np.nan)
    pval_corrected[mask] = multipletests(X_pvalue[mask], method=method,
                                         is_sorted=False,
                                         returnsorted=False)[1]
    return list(pval_corrected)


def get_fc(X_true, X_pred):
    fc = X_true / X_pred
    return fc


def get_logfc(X_true, X_pred):
    fc = np.log1p(X_true) - np.log1p(X_pred)
    return fc


def get_fc_in_logspace(X_true, X_pred):
    fc = X_true - X_pred
    return fc


def get_z_score(X_logfc):
    z_score = (X_logfc - np.nanmean(X_logfc, axis=0)) / np.nanstd(X_logfc,
                                                                  axis=0)
    return z_score


def get_ROC_AUC(X_pvalue, X_is_outlier):
    score = X_pvalue[~np.logical_or(np.isnan(X_pvalue),
                                    np.isnan(X_is_outlier))]
    label = X_is_outlier[~np.logical_or(np.isnan(X_pvalue),
                                        np.isnan(X_is_outlier))]
    label = np.invert(label != 0).astype('int')  # makes outlier 0, other 1

    if np.sum(np.abs(label)) == 0:
        warnings.warn("no injected outliers found"
                      " -> no ROC AUC calculation possible")

    fpr, tpr, _ = roc_curve(label, score)
    auc = roc_auc_score(label, score)
    return {"auc": auc, "fpr": fpr, "tpr": tpr}


def get_prec_recall(X_pvalue, X_is_outlier):
    score = -X_pvalue[~np.logical_or(np.isnan(X_pvalue),
                                     np.isnan(X_is_outlier))]
    label = X_is_outlier[~np.logical_or(np.isnan(X_pvalue),
                                        np.isnan(X_is_outlier))]

    if np.sum(np.abs(label)) == 0:
        warnings.warn("no injected outliers found"
                      " -> no precision-recall calculation possible")

    label = (label != 0).astype('int')

    pre, rec, _ = precision_recall_curve(label, score)
    curve_auc = auc(rec, pre)
    return {"auc": curve_auc, "pre": pre, "rec": rec}
