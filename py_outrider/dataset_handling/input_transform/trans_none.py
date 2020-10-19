import numpy as np
import tensorflow as tf    # 2.0.0
from tensorflow import math as tfm
import tensorflow_probability as tfp

from py_outrider.distributions.dis.dis_abstract import Dis_abstract
from py_outrider.dataset_handling.input_transform.trans_abstract import Trans_abstract
from py_outrider.utils.stats_func import multiple_testing_nan
from py_outrider.distributions.loss_dis.loss_dis_gaussian import Loss_dis_gaussian
from py_outrider.utils.stats_func import get_fc_in_logspace



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


