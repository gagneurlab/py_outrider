import tensorflow as tf
from tensorflow import math as tfm
import warnings

from ae_models.encoder_fit.E_abstract import E_abstract

from distributions.loss_dis.loss_dis_abstract import Loss_dis_abstract
from distributions.loss_dis.loss_dis_neg_bin_sf_trunc import Loss_dis_neg_bin_sf_trunc



class Loss_dis_neg_bin(Loss_dis_abstract):


    @staticmethod
    @tf.function
    def tf_loss(x, x_pred, par_meas, **kwargs):
        if par_meas is None:
            warnings.warn("calculate Dis_neg_bin par_meas theta first to get reliable loss")
            theta = tf.ones(shape=(x_pred.shape[1]), dtype=x_pred.dtype)
        else:
            theta = par_meas

        t1 = x * tfm.log(x_pred) + theta * tfm.log(theta)
        t2 = (x + theta) * tfm.log(x_pred + theta)
        t3 = tfm.lgamma(theta + x) - (tfm.lgamma(theta) + tfm.lgamma(x + 1))  # math: k! = exp(lgamma(k+1))

        ll = - tf.reduce_mean(t1 - t2 + t3)
        return ll



    @staticmethod
    @tf.function
    def tf_loss_E(e, D, b, x, x_trans, par_sample, par_meas, cov_sample, data_trans, **kwargs):
        if data_trans.trans_name=="trans_sf":
            return Loss_dis_neg_bin_sf_trunc.tf_loss_E(e=e, D=D, b=b, x=x, x_trans=x_trans,
                                             par_sample=par_sample, par_meas=par_meas, cov_sample=cov_sample, **kwargs)
        else:
            H = E_abstract.reshape_e_to_H(e=e, ae_input=x_trans, X=x, D=D, cov_sample=cov_sample)
            y = tf.matmul(H, D) + b

            x_pred = data_trans.rev_transform(y=y, par_sample=par_sample, **kwargs)
            return Loss_dis_neg_bin.tf_loss(x, x_pred, par_meas)



    @staticmethod
    @tf.function
    def tf_loss_D_single(H, x_i, b_and_D, par_sample, par_meas_i, data_trans, **kwargs):
        if data_trans.trans_name=="trans_sf":
            return Loss_dis_neg_bin_sf_trunc.tf_loss_D_single(H=H, x_i=x_i, b_and_D = b_and_D, par_sample = par_sample,
                                                           par_meas_i=par_meas_i, **kwargs)
        else:
            b_i = b_and_D[0]
            D_i = b_and_D[1:]
            y = tf.squeeze( tf.matmul(H, tf.expand_dims(D_i,1)) + b_i )

            x_pred = data_trans.rev_transform(y, par_sample=par_sample, **kwargs)
            return Loss_dis_neg_bin.tf_loss(x_i, x_pred, par_meas_i)


    @staticmethod
    @tf.function
    def tf_loss_D(H, x , b_and_D, par_sample, par_meas, data_trans, **kwargs):
        if data_trans.trans_name=="trans_sf":
            return Loss_dis_neg_bin_sf_trunc.tf_loss_D(H=H, x=x, b_and_D = b_and_D, par_sample = par_sample,
                                                           par_meas=par_meas, **kwargs)
        else:
            b_and_D = tf.reshape(b_and_D, [H.shape[1] + 1, x.shape[1]])
            b = b_and_D[0, :]
            D = b_and_D[1:, :]

            y = tf.transpose(tf.matmul(H, D) + b)

            x_pred = data_trans.rev_transform(y, par_sample=par_sample, **kwargs)
            return Loss_dis_neg_bin.tf_loss(x, x_pred, par_meas)










