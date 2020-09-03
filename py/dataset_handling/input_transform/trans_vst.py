import numpy as np
import tensorflow as tf    # 2.0.0
from tensorflow import math as tfm
import tensorflow_probability as tfp

from distributions.dis.dis_abstract import Dis_abstract
from dataset_handling.input_transform.trans_abstract import Trans_abstract
from utils.stats_func import multiple_testing_nan
from distributions.loss_dis.loss_dis_gaussian import Loss_dis_gaussian




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















