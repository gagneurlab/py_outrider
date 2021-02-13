import numpy as np
import tensorflow as tf    # 2.0.0
from tensorflow import math as tfm
import tensorflow_probability as tfp
import warnings

from .dis_abstract import Distribution
from ..fit_components.latent_space_fit.E_abstract import E_abstract
from ..utils import tf_helper_func as tfh
from ..utils.stats_func import multiple_testing_nan
from ..utils.float_limits import check_range_for_log, min_value_exp


class Dis_neg_bin(Distribution):


    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    ### pval calculation
    def get_pvalue(self):
        self.pvalue = self.tf_get_pval(self.X, self.X_pred, self.dispersions).numpy()
        return self.pvalue

    @tf.function
    def tf_get_pval(self, X, X_pred, theta):
        X_cols = tf.range(tf.shape(X)[1], dtype=tf.int32)
        pval = tf.map_fn(lambda x: (self._tf_get_pval_neg_bin(X=X[:, x], X_pred=X_pred[:, x], theta=theta[x])), X_cols,
                         dtype=X.dtype, parallel_iterations=self.parallel_iterations)
        return tf.transpose(pval)



    @classmethod
    @tf.function
    def _tf_get_pval_neg_bin(cls, X, X_pred, theta):
        return tfh.tf_nan_func(cls._tf_pval_neg_bin, X=X, X_pred=X_pred, theta=theta)


    @staticmethod
    @tf.function
    def _tf_pval_neg_bin(X, X_pred, theta):
        var = X_pred + X_pred ** 2 / theta  # variance of neg bin
        p = (var - X_pred) / var  # probabilities
        dis = tfp.distributions.NegativeBinomial(total_count=theta, probs=p)

        cum_dis_func = dis.cdf(X)
        dens_func = dis.prob(X)
        pval = 2 * tfm.minimum(tf.constant(0.5, dtype=X.dtype),
                               tfm.minimum(cum_dis_func, (1 - cum_dis_func + dens_func)))
        return pval




    ### multiple testing adjusted pvalue
    def get_pvalue_adj(self, method='fdr_by'):
        if self.pvalue is None:
            self.pvalues = self.get_pval()
        pval_adj = np.array([multiple_testing_nan(row, method=method) for row in self.pvalue])
        return pval_adj



    ### loss
    def get_loss(self):
        return self.tf_loss(x=self.X, x_pred=self.X_pred, dispersions=self.dispersions).numpy()


    @staticmethod
    def _get_random_values(inj_mean, inj_sd, size):
        raise NotImplementedError


    @staticmethod
    @tf.function
    def tf_loss(x, x_pred, dispersions, **kwargs):
        # print("Tracing with nb_tf_loss x = ", x, "\nx_pred =", x_pred, "\ndispersions =", dispersions)
        if dispersions is None:
            warnings.warn("calculate Dis_neg_bin dispersions theta first to get reliable loss")
            theta = tf.ones(shape=(x_pred.shape[1]), dtype=x_pred.dtype)
        else:
            theta = dispersions

        # handling inf / 0 values (roughly for y < -700 or y > 700)
        #x_pred = check_range_for_log(x_pred)
        
        t1 = x * tfm.log(x_pred) + theta * tfm.log(theta)
        t2 = (x + theta) * tfm.log(x_pred + theta)
        t3 = tfm.lgamma(theta + x) - (tfm.lgamma(theta) + tfm.lgamma(x + 1))  # math: k! = exp(lgamma(k+1))

        # Ignoring non-finite values in final loss
        ll = - tf.reduce_mean(tf.boolean_mask(t1 - t2 + t3, tfm.is_finite(t1 - t2 + t3)))
        return ll



    @staticmethod
    @tf.function
    def tf_loss_E( e, D, b, x, x_trans, sizefactors, dispersions, cov_sample, data_trans, **kwargs):
        if data_trans.__name__ =="Trans_sf":
                return Dis_neg_bin.tf_loss_E_trunc(e=e, D=D, b=b, x=x, x_trans=x_trans,
                                             sizefactors=sizefactors, dispersions=dispersions, cov_sample=cov_sample, **kwargs)
        else:

            return Distribution.tf_loss_E(e=e, D=D, b=b, x=x, x_trans=x_trans, sizefactors=sizefactors,
                                   dispersions=dispersions, cov_sample=cov_sample,  data_trans=data_trans, **kwargs)
            # H = E_abstract.reshape_e_to_H(e=e, ae_input=x_trans, X=x, D=D, cov_sample=cov_sample)
            # y = tf.matmul(H, D) + b
            #
            # x_pred = data_trans.rev_transform(y=y, sizefactors=sizefactors, **kwargs)
            # return Dis_neg_bin.tf_loss(x, x_pred, dispersions)



    @staticmethod
    @tf.function
    def tf_loss_D_single(H, x_i, b_and_D, sizefactors, dispersions, data_trans, **kwargs):
        if data_trans.__name__ =="Trans_sf":
            return Dis_neg_bin.tf_loss_D_single_trunc(H=H, x_i=x_i, b_and_D = b_and_D, sizefactors = sizefactors,
                                                           dispersions=dispersions, **kwargs)
        else:
            return Distribution.tf_loss_D_single(H=H, x_i=x_i, b_and_D = b_and_D, sizefactors = sizefactors,
                                                           dispersions=dispersions, data_trans=data_trans, **kwargs)
            # H, x_i, b_and_D, data_trans

            # b_i = b_and_D[0]
            # D_i = b_and_D[1:]
            # y = tf.squeeze( tf.matmul(H, tf.expand_dims(D_i,1)) + b_i )
            #
            # x_pred = data_trans.rev_transform(y, sizefactors=sizefactors, **kwargs)
            # return Dis_neg_bin.tf_loss(x_i, x_pred, dispersions_i)


    @staticmethod
    @tf.function
    def tf_loss_D(H, x , b_and_D, sizefactors, dispersions, data_trans, **kwargs):
        if data_trans.__name__ =="Trans_sf":
            return Dis_neg_bin.tf_loss_D_trunc(H=H, x=x, b_and_D = b_and_D, sizefactors = sizefactors,
                                                           dispersions=dispersions, **kwargs)
        else:
            return Distribution.tf_loss_D(H=H, x=x, b_and_D = b_and_D, sizefactors = sizefactors,
                                                           dispersions=dispersions, data_trans=data_trans, **kwargs)

            # b_and_D = tf.reshape(b_and_D, [H.shape[1] + 1, x.shape[1]])
            # b = b_and_D[0, :]
            # D = b_and_D[1:, :]
            #
            # y = tf.transpose(tf.matmul(H, D) + b)
            #
            # x_pred = data_trans.rev_transform(y, sizefactors=sizefactors, **kwargs)
            # return Dis_neg_bin.tf_loss(x, x_pred, dispersions)
            

    ### trunc loss functions
    @staticmethod
    @tf.function
    def tf_loss_E_trunc(e, D, b, x, x_trans, sizefactors, dispersions, cov_sample, **kwargs):
        sf = sizefactors
        theta = dispersions

        _, H = E_abstract.reshape_e_to_H(e=e, fit_input=x_trans, X=x, D=D, cov_sample=cov_sample)

        y = tf.matmul(H, D) + b
        y = min_value_exp(y)

        t1 = x * (tf.expand_dims(tfm.log(sf), 1) + y)
        t2 = (x + theta) * (tf.expand_dims(tfm.log(sf), 1) + y
                            + tfm.log(1 + theta / (tf.expand_dims(sf, 1) * tfm.exp(y))))
        ll = tfm.reduce_mean(t1 - t2)
        return -ll


    @staticmethod
    @tf.function
    def tf_loss_D_single_trunc(H, x_i, b_and_D, sizefactors, dispersions, **kwargs):
        sf = sizefactors
        theta_i = dispersions

        b_i = b_and_D[0]
        D_i = b_and_D[1:]
        # print("Tracing tf_loss_D_single with H = ", H, "\nx_i = ", x_i, "\nD_i = ", D_i, "\nb_i = ", b_i)
        y = tf.squeeze( tf.matmul(H, tf.expand_dims(D_i,1)) + b_i )
        y = min_value_exp(y)

        t1 = x_i * (tfm.log(sf) + y)
        t2 = (x_i + theta_i) * ( tfm.log(sf) + y + tfm.log(1 + theta_i / (sf * tfm.exp(y))) )
        ll = tfm.reduce_mean(t1-t2)
        return -ll


    @staticmethod
    @tf.function
    def tf_loss_D_trunc(H, x , b_and_D, sizefactors, dispersions, **kwargs):
        sf = sizefactors
        theta = dispersions

        b_and_D = tf.reshape(b_and_D, [H.shape[1] + 1, x.shape[1]])
        b = b_and_D[0, :]
        D = b_and_D[1:, :]

        y = tf.transpose(tf.matmul(H, D) + b)
        y = min_value_exp(y)

        t1 = tf.transpose(x) * (tfm.log(sf) + y)
        t2 = tf.transpose(x + theta) * (tfm.log(sf) + y + tfm.log(1 + tf.expand_dims(theta, 1) / (sf * tfm.exp(y))))
        ll = tf.reduce_mean(t1 - t2)
        return -ll








