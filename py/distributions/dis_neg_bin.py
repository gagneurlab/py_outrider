import numpy as np
import tensorflow as tf    # 2.0.0
from tensorflow import math as tfm
import tensorflow_probability as tfp

from distributions.dis_abstract import Dis_abstract
from distributions.tf_loss_func import tf_neg_bin_loss
from utilis.stats_func import multiple_testing_nan




class Dis_neg_bin(Dis_abstract):

    dis_name = "Dis_neg_bin"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    ### pval calculation
    def get_pvalue(self):
        self.pvalue = self.tf_get_pval(self.X_true, self.X_pred, self.par).numpy()
        return self.pvalue


    def tf_get_pval(self, X_true, X_pred, theta):
        X_true_cols = tf.range(tf.shape(X_true)[1], dtype=tf.int32)
        pval = tf.map_fn(lambda x: (self._tf_get_pval_gene(X_true[:, x], X_pred[:, x], theta[x])), X_true_cols,
                         dtype=X_true.dtype, parallel_iterations=self.parallel_iterations)
        return tf.transpose(pval)


    def _tf_get_pval_gene(self, X_true, X_pred, theta):
        var = X_pred + X_pred ** 2 / theta  # variance of neg bin
        p = (var - X_pred) / var  # probabilities
        dis = tfp.distributions.NegativeBinomial(total_count=theta, probs=p)

        cum_dis_func = dis.cdf(X_true)
        dens_func = dis.prob(X_true)
        pval = 2 * tfm.minimum(tf.constant(0.5, dtype=X_true.dtype),
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
        return tf_neg_bin_loss(self.X_true, self.X_pred, self.par).numpy()
















