import numpy as np
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc




def multiple_testing_nan(X_pvalue, method='fdr_by'):
    mask = np.isfinite(X_pvalue)
    pval_corrected = np.empty(X_pvalue.shape)
    pval_corrected.fill(np.nan)
    pval_corrected[mask] = multipletests(X_pvalue[mask], method=method, is_sorted=False, returnsorted=False)[1]
    return list(pval_corrected)




def get_log2fc(X_true, X_pred):
    fc = np.log2(X_true +1) - np.log2(X_pred+1)
    return fc

def get_log2fc_outlier(X_true, X_pred, log_value = 1.5):
    fc = get_log2fc(X_true, X_pred)
    return fc > np.log(log_value)


def get_z_score(X_log2fc):
    z_score = (X_log2fc - np.nanmean(X_log2fc, axis=0)) / np.nanstd(X_log2fc, axis=0)
    return z_score




### fpr and tpr
def get_ROC_AUC(X_pvalue, X_is_outlier):
    score = X_pvalue[~np.isnan(X_pvalue)]
    label = X_is_outlier[~np.isnan(X_is_outlier)]
    label = np.invert(label != 0).astype('int')  # makes outlier 0, other 1

    fpr, tpr, _ = roc_curve(label, score)
    auc = roc_auc_score(label, score)
    return {"auc": auc, "fpr": fpr, "tpr": tpr}


### prec and rec
def get_prec_recall(X_pvalue, X_is_outlier):
    score = -X_pvalue[~np.isnan(X_pvalue)]
    label = X_is_outlier[~np.isnan(X_is_outlier)]
    label = (label != 0).astype('int')

    pre, rec, _ = precision_recall_curve(label, score)
    curve_auc = auc(rec, pre)
    return {"auc": curve_auc, "pre": pre, "rec": rec}




