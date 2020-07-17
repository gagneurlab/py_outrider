import numpy as np
import tensorflow as tf    # 2.0.0
from tensorflow import math as tfm
import tensorflow_probability as tfp

from distributions.dis.dis_abstract import Dis_abstract
from dataset_handling.input_transform.trans_abstract import Trans_abstract
from utilis.stats_func import multiple_testing_nan
from distributions.loss_dis.loss_dis_gaussian import Loss_dis_gaussian
from utilis.stats_func import get_fc



class Trans_log(Trans_abstract):

    trans_name = "trans_log"


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
        return get_fc(X_trans, X_trans_pred)














