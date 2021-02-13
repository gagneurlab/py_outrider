import numpy as np
import tensorflow as tf    # 2.0.0
from tensorflow import math as tfm
import tensorflow_probability as tfp

from .dis_abstract import Distribution
from ..fit_components.latent_space_fit.E_abstract import E_abstract
from ..utils.stats_func import multiple_testing_nan
from ..utils import tf_helper_func as tfh



class Dis_gaussian(Distribution):


    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    ### pval calculation
    def get_pvalue(self):
        self.pvalue = self.tf_get_pval(self.X, self.X_pred).numpy()
        return self.pvalue


    @tf.function
    def tf_get_pval(self, X, X_pred):
        X_cols = tf.range(tf.shape(X)[1], dtype=tf.int32)
        pval = tf.map_fn(lambda x: (Dis_gaussian._tf_get_pval_gaus(X=X[:, x], X_pred=X_pred[:, x])), X_cols,
                         dtype=X.dtype, parallel_iterations=self.parallel_iterations)
        return tf.transpose(pval)



    @classmethod
    @tf.function
    def _tf_get_pval_gaus(cls, X, X_pred):
        return tfh.tf_nan_func(cls._tf_pval_gaus, X=X, X_pred=X_pred)

    @staticmethod
    @tf.function
    def _tf_pval_gaus(X, X_pred):
        x_res = X - X_pred
        pvalues_sd = tf.math.reduce_std(x_res)  # != R-version: ddof=1
        dis = tfp.distributions.Normal(loc=X_pred, scale=pvalues_sd)
        cdf_values = dis.cdf(X)
        pval = 2 * tfm.minimum(cdf_values, (1 - cdf_values))
        return pval



    ### multiple testing adjusted pvalue
    def get_pvalue_adj(self, method='fdr_by'):
        if self.pvalue is None:
            self.pvalues = self.get_pval()
        pval_adj = np.array([multiple_testing_nan(row, method=method) for row in self.pvalue])
        return pval_adj



    ### loss
    def get_loss(self):
        return self.tf_loss(self.X, self.X_pred).numpy()


    @staticmethod
    def _get_random_values(inj_mean, inj_sd, size):
        z_score = np.random.normal(loc=inj_mean, scale=inj_sd, size=size)
        return z_score


    @staticmethod
    @tf.function
    def tf_loss(x, x_pred, **kwargs):
        x_na = tfm.is_finite(x)
        return tf.keras.losses.MeanSquaredError()(tf.boolean_mask(x, x_na), tf.boolean_mask(x_pred, x_na))













