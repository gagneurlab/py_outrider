import numpy as np
import tensorflow as tf    # 2.0.0
from tensorflow import math as tfm
import tensorflow_probability as tfp

from py_outrider.distributions.dis.dis_abstract import Dis_abstract
from py_outrider.dataset_handling.input_transform.trans_abstract import Trans_abstract
from py_outrider.utils.stats_func import multiple_testing_nan
from py_outrider.distributions.loss_dis.loss_dis_gaussian import Loss_dis_gaussian
from py_outrider.utils.stats_func import get_logfc


class Trans_sf(Trans_abstract):


    @staticmethod
    def get_transformed_xrds(xrds):
        counts = xrds["X"].values
        sf = Trans_sf.calc_size_factor(counts)
        xrds["par_sample"] = (("sample"), sf)

        return np.log((counts + 1) /  np.expand_dims(sf,1))


    @staticmethod
    def _calc_size_factor_per_sample(gene_list, loggeomeans, counts):
        sf_sample = np.exp( np.nanmedian((np.log(gene_list) - loggeomeans)[np.logical_and(np.isfinite(loggeomeans), counts[0, :] > 0)]))
        return sf_sample

    @staticmethod
    def calc_size_factor(counts):
        loggeomeans = np.nanmean(np.log(counts), axis=0)
        sf = [Trans_sf._calc_size_factor_per_sample(x, loggeomeans, counts) for x in counts]
        return sf


    @staticmethod
    def rev_transform(y, par_sample, **kwargs):
        sf = par_sample
        if tf.is_tensor(y):
            return tfm.exp(y) * tf.expand_dims(sf, 1)
        else:
            return np.exp(y) * np.expand_dims(sf,1)



    @staticmethod
    def get_logfc(X_trans, X_trans_pred, par_sample, **kwargs):
        X = Trans_sf.rev_transform(X_trans, par_sample=par_sample)
        X_pred = Trans_sf.rev_transform(X_trans_pred, par_sample=par_sample)
        return get_logfc(X, X_pred)





