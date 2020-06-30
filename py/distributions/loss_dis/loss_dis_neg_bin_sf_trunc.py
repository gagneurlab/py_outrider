import tensorflow as tf
from tensorflow import math as tfm
from utilis.float_limits import min_value_exp

from fit_components.latent_space_fit.E_abstract import E_abstract

from distributions.loss_dis.loss_dis_abstract import Loss_dis_abstract


### only for size factor transformed data !
class Loss_dis_neg_bin_sf_trunc(Loss_dis_abstract):



    @staticmethod
    @tf.function
    def tf_loss(x, x_pred, par_meas, **kwargs):
        raise ValueError("use Loss_dis_neg_bin() class to access loss")
        # return Loss_dis_neg_bin.tf_loss(x, x_pred, par_meas, **kwargs)




    @staticmethod
    @tf.function
    def tf_loss_E(e, D, b, x, x_trans, par_sample, par_meas, cov_sample, **kwargs):
        sf = par_sample
        theta = par_meas

        _, H = E_abstract.reshape_e_to_H(e=e, ae_input=x_trans, X=x, D=D, cov_sample=cov_sample)

        y = tf.matmul(H, D) + b
        y = min_value_exp(y)

        t1 = x * (tf.expand_dims(tfm.log(sf), 1) + y)
        t2 = (x + theta) * (tf.expand_dims(tfm.log(sf), 1) + y
                            + tfm.log(1 + theta / (tf.expand_dims(sf, 1) * tfm.exp(y))))
        ll = tfm.reduce_mean(t1 - t2)
        return -ll


    @staticmethod
    @tf.function
    def tf_loss_D_single(H, x_i, b_and_D, par_sample, par_meas, **kwargs):
        sf = par_sample
        theta_i = par_meas

        b_i = b_and_D[0]
        D_i = b_and_D[1:]
        y = tf.squeeze( tf.matmul(H, tf.expand_dims(D_i,1)) + b_i )
        y = min_value_exp(y)

        t1 = x_i * (tfm.log(sf) + y)
        t2 = (x_i + theta_i) * ( tfm.log(sf) + y + tfm.log(1 + theta_i / (sf * tfm.exp(y))) )
        ll = tfm.reduce_mean(t1-t2)
        return -ll


    @staticmethod
    @tf.function
    def tf_loss_D(H, x , b_and_D, par_sample, par_meas, **kwargs):
        sf = par_sample
        theta = par_meas

        b_and_D = tf.reshape(b_and_D, [H.shape[1] + 1, x.shape[1]])
        b = b_and_D[0, :]
        D = b_and_D[1:, :]

        y = tf.transpose(tf.matmul(H, D) + b)
        y = min_value_exp(y)

        t1 = tf.transpose(x) * (tfm.log(sf) + y)
        t2 = tf.transpose(x + theta) * (tfm.log(sf) + y + tfm.log(1 + tf.expand_dims(theta, 1) / (sf * tfm.exp(y))))
        ll = tf.reduce_mean(t1 - t2)
        return -ll










