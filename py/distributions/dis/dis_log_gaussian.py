import numpy as np
import tensorflow as tf    # 2.0.0
from tensorflow import math as tfm
import tensorflow_probability as tfp

from distributions.dis.dis_abstract import Dis_abstract
from distributions.dis.dis_gaussian import Dis_gaussian
from utilis.stats_func import multiple_testing_nan
from distributions.tf_loss_func import tf_gaus_loss
from distributions.loss_dis.loss_dis_log_gaussian import Loss_dis_log_gaussian
from distributions.loss_dis.loss_dis_gaussian import Loss_dis_gaussian


#https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/LogNormal




class Dis_log_gaussian(Dis_abstract):

    dis_name = "Dis_log_gaussian"

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
        pval = tf.map_fn(lambda x: (Dis_gaussian._tf_get_pval_gene(X[:, x], X_pred[:, x])), X_cols,
                         dtype=X.dtype, parallel_iterations=self.parallel_iterations)
        return tf.transpose(pval)

    # @tf.function
    # def _tf_get_pval_gene(self, X, X_pred):
    #     x_res = X - X_pred
    #     pvalues_sd = tf.math.reduce_std(x_res)  # != R-version: ddof=1
    #     dis = tfp.distributions.LogNormal(loc=X_pred, scale=pvalues_sd)
    #     cdf_values = dis.cdf(X)
    #     pval = 2 * tfm.minimum(cdf_values, (1 - cdf_values))
    #     # pval[np.isnan(X)] = np.nan
    #     return pval



    ### multiple testing adjusted pvalue
    def get_pvalue_adj(self, method='fdr_by'):
        if self.pvalue is None:
            self.pvalues = self.get_pval()
        pval_adj = np.array([multiple_testing_nan(row, method=method) for row in self.pvalue])
        return pval_adj


    ### loss
    def get_loss(self):
        return Loss_dis_gaussian.tf_loss(self.X, self.X_pred).numpy()
        # return Loss_dis_log_gaussian.tf_loss(self.X, self.X_pred).numpy()


    @staticmethod
    def get_random_values(inj_mean, inj_sd, size):
        log_mean = np.log(inj_mean) if inj_mean != 0 else 0
        z_score = np.random.lognormal(mean=log_mean, sigma=np.log(inj_sd), size=size)
        return z_score



