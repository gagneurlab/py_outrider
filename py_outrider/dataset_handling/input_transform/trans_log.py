import numpy as np
import tensorflow as tf    # 2.0.0
from tensorflow import math as tfm
import tensorflow_probability as tfp

from py_outrider.distributions.dis.dis_abstract import Dis_abstract
from py_outrider.dataset_handling.input_transform.trans_abstract import Trans_abstract
from py_outrider.utils.stats_func import multiple_testing_nan
from py_outrider.distributions.loss_dis.loss_dis_gaussian import Loss_dis_gaussian
from py_outrider.utils.stats_func import get_logfc



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














