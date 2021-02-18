import numpy as np
from statsmodels.stats.multitest import multipletests

from .distributions import DISTRIBUTIONS
from .utils import print_func

def call_outliers(adata, 
                  distribution, 
                  fdr_method='fdr_by', 
                  effect_type='zscores',
                  num_cpus=1):
    """
    Calls outliers based on the given distribution.
    :param adata: AnnData object annotated with py_outrider model fit results.
    :param fdr_method: name of pvalue adjustment method (from statsmodels.stats.multitest.multipletests).
    :param effect_type: the type of method for effect size calculation. Must be one or a list of 'none', 'fold-change', 'zscores' or 'delta'.
    :param num_cpus: the number of cores to run p value calculation in parallel
    :return: AnnData object with pvalues, adjusted pvalues and effect sizes.
    """
    print_func.print_time('Calculating p values ...')
    distr = DISTRIBUTIONS[distribution]
    if "dispersions" in adata.varm.keys():
        dispersions=adata.varm["dispersions"]
    else:
        dispersions=None
    pvalues = distr.calc_pvalues(adata.layers["X_prepro"], 
                             adata.layers["X_predicted"], 
                             # dispersions=adata.varm["dispersions"],
                             dispersions=dispersions,
                             parallel_iterations=num_cpus)
    adata.layers["X_pvalue"] = pvalues.numpy()
    # print(f'adata.layers["X_pvalue"] without na size = {adata.layers["X_pvalue"][~np.isfinite(adata.layers["X_pvalue"])].size}')
    
    print_func.print_time('Calculating adjusted p values ...')
    adata = calc_adjusted_pvalues(adata, method=fdr_method)
    
    print_func.print_time('Calculating effect sizes ...')
    adata = calc_effect(adata, effect_type=effect_type, distribution=distr)
    return adata
    
def calc_adjusted_pvalues(adata, method='fdr_by'):
    """
    Calculates pvalues adjusted per sample with the given method.
    :param data: AnnData object annotated with model fit results.
    :param method: name of pvalue adjustment method (from statsmodels.stats.multitest.multipletests).
    :return: AnnData object with adjusted pvalues.
    """
    assert "X_pvalue" in adata.layers.keys(), 'No p-values found in AnnData object, calculate them first.'
    
    adata.layers["X_padj"] = np.array([multiple_testing_nan(row, method=method) for row in adata.layers["X_pvalue"]])
    return adata

def multiple_testing_nan(X_pvalue, method='fdr_by'):
    """
    Applies multiple testing correction while dealing with nan values.
    :param X_pvalue: array of p-values.
    :param method: name of correction method (from statsmodels.stats.multitest.multipletests).
    :return: array of corrected p-values with nan values.
    """
    mask = np.isfinite(X_pvalue)
    pval_corrected = np.empty(X_pvalue.shape)
    pval_corrected.fill(np.nan)
    pval_corrected[mask] = multipletests(X_pvalue[mask], method=method, is_sorted=False, returnsorted=False)[1]
    return list(pval_corrected)
    
def calc_effect(adata, distribution, effect_type=['fold_change', 'zscores']):
    
    if isinstance(effect_type, str):
        effect_type = [effect_type]
    for e_type in effect_type:
        assert e_type in ('fold_change', 'zscores', 'delta', 'none'), f'Unknown effect_type: {e_type}'
    
    if "fold_change" in effect_type:
        fc = (adata.layers["X_predicted"] + 1) / (adata.layers["X_prepro"] + 1)
        adata.layers["outrider_fc"] = fc
        adata.layers["outrider_l2fc"] = np.log2(fc)
        
    delta = adata.layers["X_prepro"] - adata.layers["X_predicted"]
    if "delta" in effect_type:
        adata.layers["outrider_delta"] = delta
    if "zscores" in effect_type:
        feature_means = np.nanmean(delta, axis=0)
        feature_sd = np.nanstd(delta, axis=0)
        z = (delta - feature_means) / feature_sd
        adata.layers["outrider_zscore"] = z
    
    return adata
