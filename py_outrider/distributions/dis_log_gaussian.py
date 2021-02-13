import numpy as np
import tensorflow as tf    # 2.0.0
from tensorflow import math as tfm
import tensorflow_probability as tfp

from .dis_abstract import Distribution
from .dis_gaussian import Dis_gaussian
from ..utils.stats_func import multiple_testing_nan
from ..fit_components.latent_space_fit.E_abstract import E_abstract

#https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/LogNormal




class Dis_log_gaussian(Distribution):


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.X = tfm.log1p(self.X)
        self.X_pred = tfm.log1p(self.X_pred)



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
        log_mean = np.log(inj_mean) if inj_mean != 0 else 0
        z_score = np.random.lognormal(mean=log_mean, sigma=np.log(inj_sd), size=size)
        return z_score
        
        
    # @tf.function
    @staticmethod
    def tf_loss(x, x_pred, **kwargs):
        # return tfm.log1p(tf.keras.losses.MeanSquaredError()(x, x_pred))

        # print('log_loss')
        # print(x)
        # print(x_pred)

        # tf.print(x)
        # print(x)
        # tf.print(x_pred.numpy())

        x_log = tfm.log1p(x)
        x_pred_log = tfm.log1p(x_pred)

        x_na = tfm.is_finite(x_log)
        gaus_loss = tf.keras.losses.MeanSquaredError()(tf.boolean_mask(x_log, x_na), tf.boolean_mask(x_pred_log, x_na))
        # gaus_loss = tf.keras.losses.MeanSquaredError()(x_log, x_pred_log)
        return gaus_loss



