import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import math as tfm
import anndata

from .utils import print_func
from .utils.float_limits import check_range_exp


# Preprocessing functionalities
def preprocess(adata,
               prepro_func='none',
               transformation='none',
               sf_norm=True,
               centering=True,
               noise_factor=0.0,
               covariates=None):
    assert isinstance(adata, anndata.AnnData), (
        'adata must be an AnnData instance')

    # preprocess
    print_func.print_time("Preprocessing input ...")
    adata.raw = adata
    adata = prepro(adata, prepro_func)

    # sizefactor calculation + normalization
    if sf_norm is True:
        adata = sf_normalize(adata)
    else:
        adata.obsm["sizefactors"] = np.ones(adata.n_obs)

    # transform
    adata = transform(adata, transformation)

    # add noise if requested
    adata = add_noise(adata, noise_factor)

    # centering
    if centering is True:
        adata = center(adata)

    # prepare covariates for inclusion in fit
    adata = prepare_covariates(adata, covariates)

    # put input matrix back to adata.X (AE input is in X_AE_input)
    adata.X = adata.layers["X_raw"]

    return adata


def center(adata):
    adata.varm['means'] = np.nanmean(adata.X, axis=0)
    adata.X = adata.X - adata.varm['means']
    return adata


def sf_normalize(adata):
    adata = calc_sizefactors(adata)
    sf = adata.obsm["sizefactors"]
    adata.X = adata.X / np.expand_dims(sf, 1)
    adata.layers["X_sf_norm"] = adata.X
    return adata


def calc_sizefactors(adata):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loggeomeans = np.nanmean(np.log(adata.X), axis=0)
        sf = [_calc_size_factor_per_sample(x, loggeomeans) for x in adata.X]

    adata.obsm["sizefactors"] = np.array(sf)
    return adata


def _calc_size_factor_per_sample(sample_values, loggeomeans):
    sf_sample = np.exp(np.nanmedian((np.log(sample_values) - loggeomeans)[
        np.logical_and(np.isfinite(loggeomeans), sample_values > 0)]))
    return sf_sample


def prepro(adata, prepro_func):
    adata.layers["X_raw"] = adata.X

    if isinstance(prepro_func, str):
        assert prepro_func in ('none', 'log', 'log1p', 'log2'), (
            'Unknown prepro function')

        adata.uns["prepro_func_name"] = prepro_func
        if prepro_func == 'log':
            prepro_func = np.log
        elif prepro_func == 'log1p':
            prepro_func = np.log1p
        elif prepro_func == 'log2':
            prepro_func = np.log2
        elif prepro_func == 'none':
            prepro_func = lambda x: x
        # elif prepro_func == 'vst':
        #     prepro_func = vst # TODO implement
    else:
        adata.uns["prepro_func_name"] = prepro_func.__name__

    adata.X = prepro_func(adata.X)

    adata.layers["X_prepro"] = adata.X
    # adata.uns["prepro_func"] = prepro_func
    return adata


def transform(adata, transform_func):

    assert transform_func in ('none', 'log', 'log1p'), (
        'Unknown tranformation function')
    adata.uns["transform_func"] = transform_func

    if transform_func != 'none':

        if transform_func == 'log':
            transform_func = np.log
        elif transform_func == 'log1p':
            transform_func = np.log1p

        adata.X = transform_func(adata.X)
        adata.layers["X_transformed"] = adata.X

    return adata


def reverse_transform(adata):

    assert "transform_func" in adata.uns.keys(), (
        'No tranform_func found in adata.uns')
    transform_func = adata.uns["transform_func"]

    if transform_func != 'none':
        adata.layers["X_predicted_no_trans"] = adata.layers["X_predicted"]

    adata.layers["X_predicted"] = rev_trans(adata.layers["X_predicted"],
                                            adata.obsm["sizefactors"],
                                            transform_func)

    return adata


def rev_trans(x_pred, sf, trans_func):
    assert trans_func in ('none', 'log', 'log1p'), (
        'Unknown tranformation function')

    if trans_func == 'log':
        x_pred = check_range_exp(x_pred)
        x_pred = np.exp(x_pred)
    elif trans_func == 'log1p':
        x_pred = check_range_exp(x_pred)
        x_pred = np.exp(x_pred)  # -1

    # multiply sf back (sf=1 if sf_norm=False so no effect)
    x_pred = x_pred * np.expand_dims(sf, 1)

    return x_pred


@tf.function
def rev_trans_tf(x_pred, sf, trans_func):
    assert trans_func in ('none', 'log', 'log1p'), (
        'Unknown tranformation function')

    if trans_func == 'log':
        x_pred = check_range_exp(x_pred)
        x_pred = tfm.exp(x_pred)
    elif trans_func == 'log1p':
        x_pred = check_range_exp(x_pred)
        x_pred = tfm.exp(x_pred)  # - 1

    # multiply sf back (sf=1 if sf_norm=False so no effect)
    x_pred = x_pred * tf.expand_dims(sf, 1)

    return x_pred


def prepare_covariates(adata, covariates=None):
    if covariates is not None:
        assert isinstance(adata, anndata.AnnData), (
            'adata must be an AnnData instance')
        assert isinstance(covariates, list), (
            "covariates has to be a list of strings")
        for cov in covariates:
            assert cov in adata.obs.columns, (
                f"Did not find column '{cov}' in adata.obs")

        cov_sample = adata.obs[covariates].copy()

        # transform each cov column to the respective 0|1 code
        for c in cov_sample:
            col = cov_sample[c].astype("category")
            if len(col.cat.categories) == 1:
                cov_sample.drop(c, axis=1, inplace=True, errors="ignore")
            elif len(col.cat.categories) == 2:
                only_01 = [(True
                            if x in [0, 1]
                            else False
                            for x in col.cat.categories)]
                if all(only_01) is True:
                    # print(f"only_01: {c}")
                    pass
                else:
                    # print(f"2 cat: {c}")
                    oneh = pd.get_dummies(cov_sample[c])
                    cov_sample[c] = oneh.iloc[:, 0]
            else:
                # print(f">2 cat: {c}")
                oneh = pd.get_dummies(cov_sample[c])
                oneh.columns = [c + "_" + str(x) for x in oneh.columns]
                cov_sample.drop(c, axis=1, inplace=True, errors="ignore")
                cov_sample = pd.concat([cov_sample, oneh], axis=1)

        print_func.print_time("Including given covariates as:")
        print(cov_sample.head())
        adata.uns["covariates_oneh"] = np.array(cov_sample.values,
                                                dtype=adata.X.dtype)
        adata.uns["X_AE_input"] = np.concatenate([adata.X, cov_sample.values],
                                                 axis=1)
    else:
        adata.uns["X_AE_input"] = adata.X

    return adata


def add_noise(adata, noise_factor):
    assert noise_factor >= 0, "noise_factor must be >= 0"

    if noise_factor > 0:
        # Add gaussian noise
        noise = (np.random.normal(loc=0, scale=1, size=adata.X.shape) *
                 noise_factor * np.nanstd(adata.X, ddof=1, axis=0))
        adata.X = adata.X + noise
        adata.layers["X_noise"] = noise

    return adata


def inject_outliers(adata, inj_freq=1e-3, inj_mean=3, inj_sd=1.6, **kwargs):
    # TODO implement
    adata = preprocess(adata,
                       prepro_func=kwargs["prepro_func"],
                       transformation=kwargs["data_trans"],
                       sf_norm=kwargs["sf_norm"],
                       centering=False,  # kwargs["centering"],
                       noise_factor=0.0,
                       covariates=None)

    if kwargs["data_trans"] != 'none':
        X_trans = adata.layers["X_transformed"]
    elif kwargs["sf_norm"] is True:
        X_trans = adata.layers["X_sf_norm"]
    else:
        X_trans = adata.layers["X_prepro"]

    # draw where to inject
    np.random.seed(kwargs["seed"])
    outlier_mask = np.random.choice(
                        [0., -1., 1.], size=X_trans.shape,
                        p=[1 - inj_freq, inj_freq / 2, inj_freq / 2])

    # insert with log normally distributed zscore in transformed space
    inj_zscores = _rlnorm(size=X_trans.shape, inj_mean=inj_mean, inj_sd=inj_sd)
    sd = np.nanstd(X_trans, ddof=1, axis=0)
    X_injected_trans = outlier_mask * inj_zscores * sd + X_trans

    # reverse transform to original space
    X_injected = rev_trans(X_injected_trans, sf=adata.obsm["sizefactors"],
                           trans_func=kwargs["data_trans"])

    # avoid inj outlier to be too strong
    max_outlier_value = np.nanmin(
                            [100 * np.nanmax(adata.layers["X_prepro"]),
                             np.finfo(adata.layers["X_prepro"].dtype).max])
    cond_value_too_big = X_injected > max_outlier_value
    X_injected[cond_value_too_big] = (
        adata.layers["X_prepro"][cond_value_too_big])
    outlier_mask[cond_value_too_big] = 0
    outlier_mask[~np.isfinite(adata.X)] = np.nan
    X_injected[~np.isfinite(adata.X)] = np.nan
    nr_out = np.sum(np.abs(outlier_mask[np.isfinite(outlier_mask)]))
    print_func.print_time(f"Injecting {nr_out} outliers "
                          f"(freq = {nr_out/adata.X.size})")

    # return new AnnData object with injected outliers
    adata_with_outliers = anndata.AnnData(X=X_injected,
                                          dtype=adata.X.dtype,
                                          obs=adata.obs)
    adata_with_outliers.layers["X_is_outlier"] = outlier_mask
    adata_with_outliers.layers["X_injected_zscore"] = inj_zscores
    return adata_with_outliers


def _rlnorm(size, inj_mean, inj_sd):
    log_mean = np.log(inj_mean) if inj_mean != 0 else 0
    return np.random.lognormal(mean=log_mean, sigma=np.log(inj_sd), size=size)


def get_k_most_variable_features(adata, k):
    if k is None:
        return range(adata.n_vars)

    assert isinstance(k, int), "k has to be an integer"
    assert isinstance(adata, anndata.AnnData), (
        "adata must be an AnnData instance")
    assert "X_AE_input" in adata.uns.keys(), (
        "X_AE_input needs to be in adata.uns, preprocess data first")

    feature_sd = np.nanstd(adata.uns["X_AE_input"], axis=0)
    most_var = np.argsort(-feature_sd)[:k]
    return most_var
