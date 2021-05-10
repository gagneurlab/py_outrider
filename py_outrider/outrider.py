import time
import anndata
import numpy as np
from statsmodels.stats.multitest import multipletests

from .preprocess import preprocess
from .models import Autoencoder_Model
from .distributions import DISTRIBUTIONS
from .utils import print_func


def outrider(adata,
             encod_dim,
             prepro_func='none',  # preprocessing options
             sf_norm=True,
             data_trans='none',
             centering=True,
             noise_factor=0.0,
             covariates=None,
             latent_space_model='AE',  # model parameters
             decoder_model='AE',
             dispersion_model='ML',
             loss_distribution='gaussian',
             optimizer='lbfgs',
             parallelize_decoder_by_feature=True,
             batch_size=None,
             nr_latent_space_features=None,
             num_cpus=1,
             float_type="float64",
             seed=7,
             iterations=15,  # training options
             convergence=1e-5,
             initialize=True,
             verbose=False,
             do_call_outliers=True,  # outlier calling options
             distribution='gaussian',
             fdr_method='fdr_by',
             effect_type='zscores',
             alpha=0.05,
             effect_cutoffs={}):
    """py_outrider API.

    Runs the OUTRIDER method on the input data. Uses a denoising
    autoencoder approach to fit expected values and calls outliers
    based on calculated p values.

    :param data: AnnData object
    :param prepro_func: Function or name of the function that should be
        applied to the input data before fitting the model. Currently
        only 'none' (no preprocessing in this case), 'log', 'log2' and 
        'log1p' are valid function names. If a function itself is 
        supplied, it will be applied to adata.X and has to output 
        another array.
    :param sf_norm: Boolean value indicating whether size factor
        normalization (as in DESeq2) should be applied.
    :param data_trans: Name of the function that should be applied to
        the preprocessed input data to transform for use in the model.
        Currently 'none' (no transformation in this case), 'log' or
        'log1p' are implemented.
    :param centering: Boolean value indication whether the input data
        should be centered by feature prior to model fitting.
    :param noise_factor: Controls the level of noise added to the
        input before denoising. Default: 0.0 (no noise added).
    :param covariates: Specify known covariates that should be included
        in the fit. Must be a list of column names of adata.obs
    :param encod_dim: Integer value giving the dimension of the latent
        space.
    :param latent_space_model: Name of the model for fitting the
        latent space. Possible options are 'AE' or 'PCA'.
    :param decoder_model: Name of the model for fitting the decoder.
        Possible options are 'AE' or 'PCA'.
    :param dispersion_model: Name of the model for fitting the
        dispersion parameters (if any). Supported options are 'ML'
        (maximum likelihood) or 'MoM' (method of moments).
    :param loss_distribution: Name of the distribution that should be
        used for calculating the model loss. One of 'NB', 'gaussian'
        or 'log-gaussian'.
    :param optimizer: Name of the optimizer used for fitting the model.
        Currently only 'lbfgs' is implemented.
    :param parallelize_decoder_by_feature: Boolean value indicating
        whether the decoder fit should be parallelized over the
        features. Default: True
    :param batch_size: Batch size used during model fitting. Default:
        None (updates after all samples)
    :param nr_latent_space_features: Limits the number of features used for
        fitting the latent space to the given integer k. The k most variable
        features will be selected. Default: None (using all features).
    :param num_cpus: Number of cores for parallelization.
    :param float_type: Specifies which float type should be used,
        highly advised to keep float64 which may take longer, but
        accuracy is strongly impaired by float32. Default: float64
    :param seed: Sets the seed of the random generator.
    :param iterations: Maximum number of iterations when fitting the
        model.
    :param convergence: Convergence limit to determine when to stop
        fitting.
    :param verbose: Boolean value indicating whether additional
        information should be printed during fitting.
    :param distribution: The distribution for calculating p values.
        One of 'NB', 'gaussian' or 'log-gaussian'.
    :param fdr_method: name of pvalue adjustment method (from
        statsmodels.stats.multitest.multipletests).
    :param effect_type: the type of method for effect size calculation.
        Must be one or a list of several of the following: 'none',
        'fold_change', 'zscores' or 'delta'.
    :param alpha: The significance level for calling outliers.
        Default: 0.05
    :param effect_cutoffs: Cutoffs to use on effect types for
        calling outliers. Must be a dict with keys specifying the
        effect type and values specifying the cutoff. Default: {}
    :param do_call_outliers: If False, no pvalue calculation and
        outlier calling will be done. Default: True
    :return: An AnnData object annotated with model fit results,
        pvalues, adjusted pvalues, aberrant state and effect sizes.
    """

    # check function arguments
    assert isinstance(adata, anndata.AnnData), (
        'adata must be an AnnData instance')
    assert encod_dim > 1, 'encoding_dim must be >= 2'
    if covariates is not None:
        assert isinstance(covariates, list), (
            'covariates must be specified as a list')

    # 1. preprocess and prepare data:
    adata = preprocess(adata,
                       prepro_func=prepro_func,
                       sf_norm=sf_norm,
                       transformation=data_trans,
                       centering=centering,
                       noise_factor=noise_factor,
                       covariates=covariates)

    # 2. build autoencoder model:
    model = Autoencoder_Model(
                        encoding_dim=encod_dim,
                        encoder=latent_space_model,
                        decoder=decoder_model,
                        dispersion_fit=dispersion_model,
                        loss_distribution=loss_distribution,
                        optimizer=optimizer,
                        parallelize_by_feature=parallelize_decoder_by_feature,
                        batch_size=batch_size,
                        nr_latent_space_features=nr_latent_space_features,
                        num_cpus=num_cpus,
                        seed=seed,
                        float_type=float_type,
                        verbose=verbose)

    # 3. fit model:
    time_ae_start = time.time()
    print_func.print_time('Start model fitting')
    model.fit(adata, initialize=initialize, iterations=iterations,
              convergence=convergence, verbose=verbose)
    print_func.print_time(
            'complete model fit time '
            f'{print_func.get_duration_sec(time.time() - time_ae_start)}')
    print_func.print_time(
            'model fit ended with loss: '
            f'{model.get_loss(adata)}')
    adata = model.predict(adata)

    # 4. call outliers / get (adjusted) pvalues:
    if do_call_outliers:
        adata = call_outliers(adata,
                              distribution=distribution,
                              fdr_method=fdr_method,
                              effect_type=effect_type,
                              alpha=alpha,
                              effect_cutoffs=effect_cutoffs,
                              num_cpus=num_cpus)

    return adata


def call_outliers(adata,
                  distribution,
                  fdr_method='fdr_by',
                  effect_type='zscores',
                  alpha=0.05,
                  effect_cutoffs={},
                  num_cpus=1):
    """Calls outliers based on the given distribution and fitted model.

    :param adata: AnnData object annotated with py_outrider model fit
        results.
    :param distribution: The distribution used for p value calculation
    :param fdr_method: Name of pvalue adjustment method (from
        statsmodels.stats.multitest.multipletests). Default: 'fdr_by'
    :param effect_type: The type of method for effect size calculation.
        Must be one or a list of several of the following: 'none',
        'fold-change', 'zscores' or 'delta'.
    :param alpha: The significance level for calling outliers.
        Default: 0.05
    :param effect_cutoffs: Cutoffs to use on effect types for
        calling outliers. Must be a dict with keys specifying the
        effect type and values specifying the cutoff. Default: {}
    :param num_cpus: Number of cores to run p value calculation in
        parallel.
    :return: AnnData object with pvalues, adjusted pvalues and effect
        sizes.
    """

    distr = DISTRIBUTIONS[distribution]
    if "dispersions" in adata.varm.keys():
        dispersions = adata.varm["dispersions"]
    else:
        dispersions = None

    print_func.print_time(f'Calculating {distr.__name__} p values ...')
    pvalues = distr.calc_pvalues(adata.layers["X_prepro"],
                                 adata.layers["X_predicted"],
                                 dispersions=dispersions,
                                 parallel_iterations=num_cpus)
    adata.layers["X_pvalue"] = pvalues.numpy()

    print_func.print_time('Calculating adjusted p values ...')
    adata = calc_adjusted_pvalues(adata, method=fdr_method)

    print_func.print_time('Calculating effect sizes ...')
    adata = calc_effect(adata, effect_type=effect_type)

    print_func.print_time('Detecting aberrant events ...')
    adata = aberrant(adata, alpha=alpha, effect_cutoffs=effect_cutoffs)

    return adata


def calc_adjusted_pvalues(adata, method='fdr_by'):
    """Calculates pvalues adjusted per sample with the given method.

    :param data: AnnData object annotated with model fit results.
    :param method: Name of pvalue adjustment method (from
        statsmodels.stats.multitest.multipletests).
    :return: AnnData object with adjusted pvalues.
    """
    assert "X_pvalue" in adata.layers.keys(), (
        'No X_pvalue found in AnnData object, calculate pvalues first.')

    adata.layers["X_padj"] = (np.array([multiple_testing_nan(row,
                                                             method=method)
                                        for row in adata.layers["X_pvalue"]]))
    return adata


def multiple_testing_nan(X_pvalue, method='fdr_by'):
    """Multiple testing correction while dealing with nan values.

    Applies multiple testing correction while dealing with nan
    values.

    :param X_pvalue: array of p-values.
    :param method: name of correction method (from
        statsmodels.stats.multitest.multipletests).
    :return: array of corrected p-values with nan values.
    """
    mask = np.isfinite(X_pvalue)
    pval_corrected = np.empty(X_pvalue.shape)
    pval_corrected.fill(np.nan)
    pval_corrected[mask] = multipletests(X_pvalue[mask], method=method,
                                         is_sorted=False,
                                         returnsorted=False)[1]
    return list(pval_corrected)


def calc_effect(adata, effect_type=['fold_change', 'zscores']):
    """Calculates effect sizes based on fitted expected values

    Calculates effect sizes ((log) fold-changes, zscores, delta values)
    based on fitted expected values.

    :param effect_type: The type of method for effect size calculation.
        Must be one or a list of several of the following: 'none',
        'fold-change', 'zscores' or 'delta'.
    :return: AnnData object annotated with specified effect types
        (stored in layers["outrider_fc/l2fc/zscore/delta"] for each
        type).
    """

    if isinstance(effect_type, str):
        effect_type = [effect_type]
    for e_type in effect_type:
        assert e_type in ('fold_change', 'zscores', 'delta', 'none'), (
            f'Unknown effect_type: {e_type}')

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


def aberrant(adata, alpha, effect_cutoffs={}):

    assert 0 < alpha <= 1, f'alpha {alpha} is not between 0 and 1'
    assert "X_padj" in adata.layers.keys(), (
        'No X_padj found in AnnData object, calculate adjusted pvalues first.')

    aberrant = adata.layers["X_padj"] < alpha

    for effect_type in effect_cutoffs:
        assert "outrider_" + effect_type in adata.layers.keys(), (
            'Did not find {"outrider_" + effect_type} in adata.layers.keys()')
        aberrant = np.logical_and(
            aberrant,
            (adata.layers["outrider_" + effect_type] > effect_cutoffs[
                                                                effect_type]))

    adata.layers["aberrant"] = aberrant

    return adata
