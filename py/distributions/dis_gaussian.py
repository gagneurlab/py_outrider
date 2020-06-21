import numpy as np
import tensorflow as tf    # 2.0.0
from tensorflow import math as tfm
import tensorflow_probability as tfp

from distributions.dis_abstract import Dis_abstract
from utilis.stats_func import multiple_testing_nan
from distributions.tf_loss_func import tf_gaus_loss



class Dis_gaussian(Dis_abstract):

    dis_name = "Dis_gaussian"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    ### pval calculation
    def get_pvalue(self):
        self.pvalue = self.tf_get_pval(self.X_true, self.X_pred).numpy()
        return self.pvalue

    @tf.function
    def tf_get_pval(self, x_true, x_pred):
        x_true_cols = tf.range(tf.shape(x_true)[1], dtype=tf.int32)
        pval = tf.map_fn(lambda x: (self._tf_get_pval_gene(x_true[:, x], x_pred[:, x])), x_true_cols,
                         dtype=x_true.dtype, parallel_iterations=self.parallel_iterations)
        return tf.transpose(pval)

    @tf.function
    def _tf_get_pval_gene(self, x_true, x_pred):
        x_res = x_true - x_pred
        pvalues_sd = tf.math.reduce_std(x_res)  # != R-version: ddof=1
        dis = tfp.distributions.Normal(loc=x_pred, scale=pvalues_sd)
        cdf_values = dis.cdf(x_true)
        pval = 2 * tfm.minimum(cdf_values, (1 - cdf_values))
        # pval[np.isnan(x_true)] = np.nan
        return pval



    ### multiple testing adjusted pvalue
    def get_pvalue_adj(self, method='fdr_by'):
        if self.pvalue is None:
            self.pvalues = self.get_pval()
        pval_adj = np.array([multiple_testing_nan(row, method=method) for row in self.pvalue])
        return pval_adj



    ### loss
    def get_loss(self):
        return tf_gaus_loss(self.X_true, self.X_pred).numpy()
















