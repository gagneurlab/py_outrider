from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow import math as tfm
import tensorflow_probability as tfp
import warnings

from ..utils.stats_func import multiple_testing_nan
from ..utils.stats_func import get_logfc, get_fc_in_logspace
from ..utils.float_limits import check_range_exp, check_range_for_log


class Trans_abstract(ABC):

    @staticmethod
    @abstractmethod
    def get_transformed_xrds(self):
        pass

    @staticmethod
    @abstractmethod
    def rev_transform(self):
        pass

    @staticmethod
    @abstractmethod
    def get_logfc(X_trans, X_trans_pred, **kwargs):
        pass
    
    @staticmethod
    @abstractmethod
    def check_range_trans(x):
        pass
    
    @staticmethod
    @abstractmethod
    def check_range_rev_trans(x):
        pass

    @classmethod
    def transform_xrds(cls, xrds):
        X_trans = cls.get_transformed_xrds(xrds)

        ### center to zero
        X_bias = np.nanmean(X_trans, axis=0)
        X_trans_cen = X_trans - X_bias

        xrds["X_trans"] = (('sample', 'meas'), X_trans_cen)
        xrds["X_center_bias"] = (('meas'), X_bias)



class Trans_log(Trans_abstract):


    @staticmethod
    def get_transformed_xrds(xrds):
        return np.log1p(xrds["X"])


    @staticmethod
    def rev_transform(y, **kwargs):
        if tf.is_tensor(y):
            return tfm.exp(y) - 1
        else:
            return np.exp(y) - 1

        # if tf.is_tensor(y):
        #     return tfm.pow(y, 2)
        # else:
        #     return np.power(y, 2)

    @staticmethod
    def get_logfc(X_trans, X_trans_pred, **kwargs):
        X = Trans_log.rev_transform(X_trans)
        X_pred = Trans_log.rev_transform(X_trans_pred)
        return get_logfc(X, X_pred)
        
    @staticmethod
    def check_range_trans(x):
        return check_range_exp(x)
        
    @staticmethod
    def check_range_rev_trans(x):
        return check_range_for_log(x)


class Trans_none(Trans_abstract):

    @staticmethod
    def get_transformed_xrds(xrds):
        return xrds["X"]



    @staticmethod
    def rev_transform(y, **kwargs):
        return y


    ### trans_none assumes that data is already in log-space, needed for profile_protrider
    @staticmethod
    def get_logfc(X_trans, X_trans_pred, **kwargs):
        return get_fc_in_logspace(X_trans, X_trans_pred)

        
    @staticmethod
    def check_range_trans(x):
        return x
        
    @staticmethod
    def check_range_rev_trans(x):
        return x
        
class Trans_sf(Trans_abstract):


    @staticmethod
    def get_transformed_xrds(xrds):
        counts = xrds["X"].values
        sf = Trans_sf.calc_size_factor(counts)
        xrds["sizefactors"] = (("sample"), sf)

        return np.log((counts + 1) /  np.expand_dims(sf,1))


    @staticmethod
    def _calc_size_factor_per_sample(sample_cnts, loggeomeans):
        sf_sample = np.exp( np.nanmedian((np.log(sample_cnts) - loggeomeans)[np.logical_and(np.isfinite(loggeomeans), sample_cnts > 0)]))
        return sf_sample

    @staticmethod
    def calc_size_factor(counts):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loggeomeans = np.nanmean(np.log(counts), axis=0)
            sf = [Trans_sf._calc_size_factor_per_sample(x, loggeomeans) for x in counts]
        return sf


    @staticmethod
    def rev_transform(y, sizefactors, **kwargs):
        sf = sizefactors
        if tf.is_tensor(y):
            return tfm.exp(y) * tf.expand_dims(sf, 1)
        else:
            return np.exp(y) * np.expand_dims(sf,1)



    @staticmethod
    def get_logfc(X_trans, X_trans_pred, sizefactors, **kwargs):
        X = Trans_sf.rev_transform(X_trans, sizefactors=sizefactors)
        X_pred = Trans_sf.rev_transform(X_trans_pred, sizefactors=sizefactors)
        return get_logfc(X, X_pred)
        
    @staticmethod
    def check_range_trans(x):
        return check_range_exp(x)
        
    @staticmethod
    def check_range_rev_trans(x):
        return check_range_for_log(x)
        

class Trans_vst(Trans_abstract):


    @staticmethod
    def get_transformed_xrds(xrds):
        ## perform vst
        ## safe parameters in xrds.attrs["other_par"] TODO implement
        return xrds["X"]


    @staticmethod
    def rev_transform(y, other_par, **kwargs):
        ### implement
        # extra_pois, asympt_disp = other_par
        # matrix_inverse_vst(y, extra_pois, asympt_disp)

        if tf.is_tensor(y):
            return y
        else:
            return y



    @staticmethod
    def get_logfc(X_trans, X_trans_pred, other_par, **kwargs):
        pass

        
    @staticmethod
    def check_range_trans(x):
        return x
        
    @staticmethod
    def check_range_rev_trans(x):
        return x


    ### TODO CHANGE TO FLEXIBLY TAKE PARAMETER
    # def matrix_inverse_vst(matrix, extraPois, asymptDisp):
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
        


