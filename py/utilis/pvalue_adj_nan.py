import numpy as np
from statsmodels.stats.multitest import multipletests




def multiple_testing_nan(pvalues, method='fdr_by'):
    mask = np.isfinite(pvalues)
    pval_corrected = np.empty(pvalues.shape)
    pval_corrected.fill(np.nan)
    pval_corrected[mask] = multipletests(pvalues[mask], method=method, is_sorted=False, returnsorted=False)[1]
    return list(pval_corrected)