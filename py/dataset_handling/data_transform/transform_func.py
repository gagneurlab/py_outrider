import numpy as np
import tensorflow as tf
from tensorflow import math as tfm


def xrds_transform(xrds):
    trans_name = xrds.attrs["profile"].ae_input_trans
    
    if trans_name == "sf":
        X_trans = transform_sf(xrds)
    elif trans_name == "log2":
        X_trans = transform_log2(xrds)
    elif trans_name == "none":
        X_trans = transform_none(xrds)

    ### center to zero
    X_bias = np.mean(X_trans, axis=0)
    X_trans_cen = X_trans - X_bias
    
    xrds["X_trans"] = (('sample', 'meas'), X_trans_cen)
    xrds["X_center_bias"] = (('meas'), X_bias)



def transform_none(xrds):
    return xrds["X"]


def transform_log2(xrds):
    return np.log2(xrds["X"]+1)


def transform_sf(xrds):
    counts = xrds["X"].values
    sf = calc_size_factor(counts)
    xrds["par_sample"] = (("sample"), sf)
    return np.log((counts + 1) / sf)


def _calc_size_factor_per_sample(gene_list, loggeomeans, counts):
    sf_sample = np.exp( np.median((np.log(gene_list) - loggeomeans)[np.logical_and(np.isfinite(loggeomeans), counts[0, :] > 0)]))
    return sf_sample


def calc_size_factor(counts):
    loggeomeans = np.mean(np.log(counts), axis=0)
    sf = [_calc_size_factor_per_sample(x, loggeomeans, counts) for x in counts]
    return sf





############################################


def rev_transform_ae_input(y, norm_name, **kwargs):
    if norm_name == "sf":
        return rev_transform_sf(y, **kwargs)
    elif norm_name == "log2":
        return rev_transform_log2(y)
    elif norm_name == "none":
        return rev_transform_none(y)


def rev_transform_none(y):
    return y


def rev_transform_log2(y):
    if tf.is_tensor(y):
        return tfm.pow(y,2)
    else:
        return np.power(y,2)


def rev_transform_sf(y, par_sample, **kwargs):
    sf = par_sample
    if tf.is_tensor(y):
        return tfm.exp(y) * tf.expand_dims(sf,1)
    else:
        return np.exp(y) * sf













#
#
# # def matrix_inverse_vst(matrix, extraPois, asymptDisp):
# @tf.function
# def matrix_inverse_vst(matrix):
#     # return tfm.exp(matrix)
#
#     extraPois = tf.constant(5.30561, dtype=matrix.dtype)  # only for mofa_outrider/06_sample_blood_outlier_z3/counts_raw.csv
#     asymptDisp = tf.constant(0.09147687, dtype=matrix.dtype)
#
#     two = tf.constant(2, dtype=matrix.dtype)
#
#     vst_inv = tfm.pow( (4 * asymptDisp * tfm.pow(two, matrix) -(1 + extraPois)), 2) / \
#               (4 * asymptDisp * (1 + extraPois + (4 * asymptDisp * tfm.pow(two,matrix) - (1 + extraPois))))
#     return vst_inv
#
#
# # def matrix_vst(matrix, extraPois, asymptDisp):
# @tf.function
# def matrix_vst(matrix):
#     extraPois = tf.constant(5.30561, dtype=matrix.dtype)
#     asymptDisp = tf.constant(0.09147687, dtype=matrix.dtype)
#
#     vst = tfm.log((1 + extraPois + 2 * asymptDisp * matrix + 2 * tfm.sqrt(asymptDisp * matrix * (1 + extraPois + asymptDisp * matrix))) /
#                   (4 * asymptDisp)) / tfm.log(tf.constant(2, dtype=matrix.dtype))
#     return vst
#

