import numpy as np
import tensorflow as tf
from tensorflow import math as tfm


def normalize_ae_input(xrds, norm_name):
    if norm_name == "sf":
        xrds_normalize_sf(xrds)
    elif norm_name == "log2":
        xrds_normalize_log2(xrds)
    elif norm_name == "none":
        xrds_normalize_none(xrds)


def xrds_normalize_none(xrds):

    def normalize_none(counts):
        x0_bias = np.mean(counts, axis=0)
        x1 = counts - x0_bias
        return x1, x0_bias

    counts, bias = normalize_none(xrds["X"])
    xrds["X_norm"] = ( ('sample','meas'), counts)
    xrds["X_center_bias"] = ( ('meas'), bias)


def xrds_normalize_log2(xrds):

    def normalize_log2(counts):
        x0 = np.log2(counts+1)
        x0_bias = np.mean(x0, axis=0)
        x1 = x0 - x0_bias
        return x1, x0_bias

    counts, bias = normalize_log2(xrds["X"])
    xrds["X_norm"] = ( ('sample','meas'), counts)
    xrds["X_center_bias"] = ( ('meas'), bias)




def xrds_normalize_sf(xrds):
    sf = calc_size_factor(xrds["X"])
    xrds["par_sample"] = (("sample"), sf)

    def normalize_sf(counts, sf):
        x0 = np.log((counts + 1) / sf )
        x0_bias = np.mean(x0, axis=0)
        x1 = x0 - x0_bias
        return x1, x0_bias

    counts, bias = normalize_sf(xrds["X"], xrds["par_sample"])
    xrds["X_norm"] = ( ('sample','meas'), counts)
    xrds["X_center_bias"] = ( ('meas'), bias)


def _calc_size_factor_per_sample(gene_list, loggeomeans, counts):
    sf_sample = np.exp( np.median((np.log(gene_list) - loggeomeans)[np.logical_and(np.isfinite(loggeomeans), counts[0, :] > 0)]))
    return sf_sample


def calc_size_factor(counts):
    count_matrix = counts.values
    loggeomeans = np.mean(np.log(count_matrix), axis=0)
    sf = [_calc_size_factor_per_sample(x, loggeomeans, count_matrix) for x in count_matrix]
    return sf





############################################


def rev_normalize_ae_input(y, norm_name, **kwargs):
    if norm_name == "sf":
        return rev_normalize_sf(y, **kwargs)
    elif norm_name == "log2":
        return rev_normalize_log2(y)
    elif norm_name == "none":
        return rev_normalize_none(y)


def rev_normalize_none(y):
    return y

def rev_normalize_log2(y):
    if tf.is_tensor(y):
        return tfm.pow(y,2)
    else:
        return np.power(y,2)

def rev_normalize_sf(y, sf):
    if tf.is_tensor(y):
        return tfm.exp(y) * tf.expand_dims(sf,1)
    else:
        return np.exp(y) * sf















