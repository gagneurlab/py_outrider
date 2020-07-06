import numpy as np
import tensorflow as tf    # 2.0.0
from tensorflow import math as tfm
import tensorflow_probability as tfp

from distributions.dis.dis_abstract import Dis_abstract
from dataset_handling.input_transform.trans_abstract import Trans_abstract
from utilis.stats_func import multiple_testing_nan
from distributions.loss_dis.loss_dis_gaussian import Loss_dis_gaussian




class Trans_sf(Trans_abstract):

    trans_name = "trans_sf"


    @staticmethod
    def get_transformed_xrds(xrds):
        counts = xrds["X"].values
        sf = Trans_sf.calc_size_factor(counts)
        xrds["par_sample"] = (("sample"), sf)

        return np.log((counts + 1) /  np.expand_dims(sf,1))


    @staticmethod
    def _calc_size_factor_per_sample(gene_list, loggeomeans, counts):
        sf_sample = np.exp( np.median((np.log(gene_list) - loggeomeans)[np.logical_and(np.isfinite(loggeomeans), counts[0, :] > 0)]))
        return sf_sample

    @staticmethod
    def calc_size_factor(counts):
        loggeomeans = np.mean(np.log(counts), axis=0)
        sf = [Trans_sf._calc_size_factor_per_sample(x, loggeomeans, counts) for x in counts]
        return sf


    @staticmethod
    def rev_transform(y, par_sample, **kwargs):
        sf = par_sample
        if tf.is_tensor(y):
            return tfm.exp(y) * tf.expand_dims(sf, 1)
        else:
            return np.exp(y) * np.expand_dims(sf,1)




