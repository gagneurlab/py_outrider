import numpy as np
import tensorflow as tf    # 2.0.0
from tensorflow import math as tfm
import tensorflow_probability as tfp

from distributions.dis.dis_abstract import Dis_abstract
from dataset_handling.data_transform.trans_abstract import Trans_abstract
from utilis.stats_func import multiple_testing_nan
from distributions.loss_dis.loss_dis_gaussian import Loss_dis_gaussian




class Trans_none(Trans_abstract):

    trans_name = "trans_none"

    @staticmethod
    def get_transformed_xrds(xrds):
        return xrds["X"]



    @staticmethod
    def rev_transform(y):
        return y



