import numpy as np
import tensorflow as tf    # 2.0.0
from tensorflow import math as tfm
import tensorflow_probability as tfp

from distributions.dis.dis_abstract import Dis_abstract
# from distributions.tf_loss_func import tf_neg_bin_loss
from utilis.stats_func import multiple_testing_nan
from distributions.loss_dis.loss_dis_neg_bin import Loss_dis_neg_bin
import utilis.tf_helper_func as tfh




class Dis_neg_bin(Dis_abstract):

    dis_name = "Dis_neg_bin"


    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    ### pval calculation
    def get_pvalue(self):
        self.pvalue = self.tf_get_pval(self.X, self.X_pred, self.par_meas).numpy()
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
        return tfh.tf_nan_func(cls._tf_pval_gaus, X=X, X_pred=X_pred, theta=theta)


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
        return Loss_dis_neg_bin.tf_loss(x=self.X, x_pred=self.X_pred, par_meas=self.par_meas).numpy()


    @staticmethod
    def get_random_values(inj_mean, inj_sd, size):
        raise NotImplementedError










