import warnings
import numpy as np
from tensorflow import math as tfm

### Preprocessing functionality
def preprocess(adata, 
               prepro_func='none',
               transformation='none',
               sf_norm=True,
               centering=True, 
               noise_factor=0.0):
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
    assert transform_func in ('none', 'log', 'log1p'), 'Unknown tranformation function'
    
    if transform_func != 'none':
        adata.layers["X_predicted_no_trans"] = adata.layers["X_predicted"]
        
    if transform_func == 'log':
        adata.layers["X_predicted"] = tfm.exp(adata.layers["X_predicted"])
    elif transform_func == 'log1p':
        adata.layers["X_predicted"] = tfm.exp(adata.layers["X_predicted"]) - 1
        
    # multiply sf back (sf=1 if sf_norm=False so no effect)
    sf = adata.obsm["sizefactors"]
    adata.layers["X_predicted"] = adata.layers["X_predicted"] * np.expand_dims(sf, 1)
        
    return adata
    
def add_noise(adata, noise_factor):
    # TODO implement
    return adata

def inject_outliers(adata, inj_freq=1e-3, inj_mean=3, inj_sd=1.6):
    # TODO implement
    return adata
