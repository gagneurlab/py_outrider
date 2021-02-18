import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import math as tfm

from .utils import print_func

### Preprocessing functionalities
def preprocess(adata, 
               prepro_func='none',
               transformation='none',
               sf_norm=True,
               centering=True, 
               noise_factor=0.0,
               covariates=None):
    # store previous object in raw
    adata.raw = adata
    
    # preprocess
    adata = prepro(adata, prepro_func)
    
    # sizefactor calculation + normalization
    if sf_norm is True:
        adata = sf_normalize(adata)
    else:
        adata.obsm["sizefactors"] = np.ones(adata.n_obs)
    
    # transform
    adata = transform(adata, transformation) 
    
    # centering
    if centering is True:
        adata = center(adata)
        
    # add noise if requested
    adata = add_noise(adata, noise_factor)
    
    # prepare covariates for inclusion in fit
    adata = prepare_covariates(adata, covariates)
        
    return adata
        
def center(adata):
    adata.varm['means'] = np.nanmean(adata.X, axis=0)
    adata.layers["X_uncentered"] = adata.X
    adata.X = adata.X - adata.varm['means']
    return adata
    
def sf_normalize(adata):
    adata = calc_sizefactors(adata)
    sf = adata.obsm["sizefactors"]
    adata.X = adata.X / np.expand_dims(sf, 1)
    return adata

def calc_sizefactors(adata):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loggeomeans = np.nanmean(np.log(adata.X), axis=0)
        sf = [_calc_size_factor_per_sample(x, loggeomeans) for x in adata.X]
    
    adata.obsm["sizefactors"] = np.array(sf)
    return adata
    
def _calc_size_factor_per_sample(sample_values, loggeomeans):
    sf_sample = np.exp( np.nanmedian((np.log(sample_values) - loggeomeans)[np.logical_and(np.isfinite(loggeomeans), sample_values > 0)]))
    return sf_sample

def prepro(adata, prepro_func):
    assert prepro_func in ('none', 'log', 'log1p', 'vst'), 'Unknown prepro function'
    adata.layers["X_raw"] = adata.X
        
    if prepro_func == 'log':
        adata.X = np.log(adata.X)
    elif prepro_func == 'log1p':
        adata.X = np.log1p(adata.X)
    elif prepro_func == 'vst':
        adata.X = vst(adata.X) # TODO implement
        
    adata.layers["X_prepro"] = adata.X
    return adata
    
def transform(adata, transform_func):
    assert transform_func in ('none', 'log', 'log1p'), 'Unknown tranformation function'
    
    if transform_func != 'none':
        adata.layers["X_no_trans"] = adata.X
        
    if transform_func == 'log':
        adata.X = np.log(adata.X)
    elif transform_func == 'log1p':
        adata.X = np.log1p(adata.X)
        
    adata.uns["transform_func"] = transform_func
    
    return adata
    
def reverse_transform(adata):
    
    assert "transform_func" in adata.uns.keys(), 'No tranform_func found in adata.uns'
    transform_func = adata.uns["transform_func"]
    
    if transform_func != 'none':
        adata.layers["X_predicted_no_trans"] = adata.layers["X_predicted"]
        
    adata.layers["X_predicted"] = rev_trans(adata.layers["X_predicted"], adata.obsm["sizefactors"], transform_func)
    
    return adata
    
@tf.function
def rev_trans(x_pred, sf, trans_func):
    assert trans_func in ('none', 'log', 'log1p'), 'Unknown tranformation function'
    
    if trans_func == 'log':
        x_pred = tfm.exp(x_pred)
    elif trans_func == 'log1p':
        x_pred = tfm.exp(x_pred) - 1
        
    # multiply sf back (sf=1 if sf_norm=False so no effect)
    x_pred = x_pred * tf.expand_dims(sf, 1)
    
    return x_pred
    
def add_noise(adata, noise_factor):
    # TODO implement
    return adata

def prepare_covariates(adata, covariates=None):
    
    if covariates is not None:
        assert isinstance(covariates, list), "covariates has to be a list of strings"
        one_hot_encoding = []
        for cov in covariates:
            assert cov in adata.obs.columns, f"Did not find column '{cov}' in adata.obs"
        
        cov_sample = adata.obs[covariates].copy()

        ### transform each cov column to the respective 0|1 code
        for c in cov_sample:
            col = cov_sample[c].astype("category")
            if len(col.cat.categories) == 1:
                cov_sample.drop(c, axis=1, inplace=True, errors="ignore")
            elif len(col.cat.categories) == 2:
                only_01 = [True if x in [0, 1] else False for x in col.cat.categories]
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
        print(cov_sample)
        adata.uns["covariates_oneh"] = np.array(cov_sample.values, dtype=adata.X.dtype)
        adata.uns["X_with_cov"] = np.concatenate([adata.X, cov_sample.values], axis=1)  
    
    return adata

def inject_outliers(adata, inj_freq=1e-3, inj_mean=3, inj_sd=1.6):
    # TODO implement
    return adata
