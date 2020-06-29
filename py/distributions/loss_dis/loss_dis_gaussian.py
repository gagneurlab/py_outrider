import tensorflow as tf
from tensorflow import math as tfm
from utilis.float_limits import min_value_exp

from ae_models.encoder_fit.E_abstract import E_abstract

from distributions.loss_dis import loss_dis_abstract
from distributions.transform_func import rev_transform_ae_input




class Loss_dis_gaussian(loss_dis_abstract):



    @staticmethod
    @tf.function
    def  tf_loss(x, x_pred, **kwargs):
        return tf.keras.losses.MeanSquaredError()(x, x_pred)

    @staticmethod
    @tf.function
    def tf_loss_E(e, D, b, x, x_trans, cov_sample, **kwargs):
        H = E_abstract.reshape_e_to_H(e=e, ae_input=x_trans, X=x, D=D, cov_sample=cov_sample)
        x_pred = tf.matmul(H, D) + b
        return Loss_dis_gaussian.tf_loss(x, x_pred)


    @staticmethod
    @tf.function
    def tf_loss_D_single(H, x_i, b_and_D, **kwargs):
        b_i = b_and_D[0]
        D_i = b_and_D[1:]
        x_pred = tf.squeeze( tf.matmul(H, tf.expand_dims(D_i,1)) + b_i )
        return Loss_dis_gaussian.tf_loss(x_i, x_pred)


    @staticmethod
    @tf.function
    def tf_loss_D(H, x , b_and_D, **kwargs):
        b_and_D = tf.reshape(b_and_D, [H.shape[1] + 1, x.shape[1]])
        b = b_and_D[0, :]
        D = b_and_D[1:, :]

        x_pred = tf.transpose(tf.matmul(H, D) + b)
        return Loss_dis_gaussian.tf_loss(x, x_pred)



















