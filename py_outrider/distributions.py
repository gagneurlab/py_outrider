from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import math as tfm

from .utils import tf_helper_func as tfh
# from .utils.float_limits import check_range_for_log, min_value_exp
from .utils.np_mom_theta import trim_mean


class Distribution(ABC):

    def __init__(self, **kwargs):
        self.distr_name = None

    @classmethod
    @abstractmethod
    def calc_pvalues(cls, x_true, x_pred, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def loss(x_true, x_pred, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def has_dispersion():
        pass

    # method of moments
    @staticmethod
    @abstractmethod
    def mom(x):
        pass


# NegativeBinomial distribution
class NB(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.distr_name = "NB"

    @staticmethod
    def has_dispersion():
        return True

    @staticmethod
    def loss(x_true, x_pred, dispersions, **kwargs):
        # print("Tracing with nb_tf_loss x = ", x, "\nx_pred =", x_pred,
        #       "\ndispersions =", dispersions)
        # if dispersions is None:
        #     warnings.warn(
        #         "NB loss: calculate dispersions first to get reliable loss")
        #     theta = tf.ones(shape=(x_pred.shape[1]), dtype=x_pred.dtype)
        # else:
        theta = dispersions

        # handling inf / 0 values (roughly for y < -700 or y > 700)
        # x_pred = check_range_for_log(x_pred)
        # print(f"NB loss: min x_pred: {tf.reduce_min(x_pred)}")
        # print(f"NB loss: max x_pred: {tf.reduce_max(x_pred)}")

        t1 = x_true * tfm.log(x_pred) + theta * tfm.log(theta)
        t2 = (x_true + theta) * tfm.log(x_pred + theta)
        t3 = tfm.lgamma(theta + x_true) - (tfm.lgamma(theta) +
                                           tfm.lgamma(x_true + 1))

        ll = - tf.reduce_mean(t1 - t2 + t3)
        # Ignoring non-finite values in final loss
        # ll = - tf.reduce_mean(tf.boolean_mask(t1 - t2 + t3,
        #                                       tfm.is_finite(t1 - t2 + t3)))
        return ll

    # robust method of moments to estimate theta
    @staticmethod
    def mom(x, theta_min=1e-2, theta_max=1e3, mu_min=0.1):
        mue = trim_mean(x, proportiontocut=0.125, axis=0)
        mue = np.maximum(mue, mu_min)

        se = (x - np.array([mue] * x.shape[0])) ** 2
        see = trim_mean(se, proportiontocut=0.125, axis=0)
        ve = 1.51 * see

        th = mue ** 2 / (ve - mue)
        th[th < 0] = theta_max

        re_th = np.maximum(theta_min, np.minimum(theta_max, th))
        return re_th

    @classmethod
    def calc_pvalues(cls, x_true, x_pred, dispersions, parallel_iterations=1):
        pvalues = cls.tf_get_pval(x_true, x_pred, dispersions,
                                  parallel_iterations=parallel_iterations)
        return pvalues

    @classmethod
    @tf.function
    def tf_get_pval(cls, X, X_pred, theta, parallel_iterations):
        X_cols = tf.range(tf.shape(X)[1], dtype=tf.int32)
        pval = tf.map_fn(lambda x: (cls._tf_get_pval_neg_bin(
                                        X=X[:, x], X_pred=X_pred[:, x],
                                        theta=theta[x])),
                         X_cols,
                         fn_output_signature=X.dtype,
                         # dtype=X.dtype, # deprecated
                         parallel_iterations=parallel_iterations)
        return tf.transpose(pval)

    @classmethod
    @tf.function
    def _tf_get_pval_neg_bin(cls, X, X_pred, theta):
        return tfh.tf_nan_func(cls._tf_pval_neg_bin, X=X, X_pred=X_pred,
                               theta=theta)

    @staticmethod
    @tf.function
    def _tf_pval_neg_bin(X, X_pred, theta):
        var = X_pred + X_pred ** 2 / theta  # variance of neg bin
        p = (var - X_pred) / var  # probabilities
        dis = tfp.distributions.NegativeBinomial(total_count=theta, probs=p)

        cum_dis_func = dis.cdf(X)
        dens_func = dis.prob(X)
        pval = 2 * tfm.minimum(tf.constant(0.5, dtype=X.dtype),
                               tfm.minimum(cum_dis_func,
                                           (1 - cum_dis_func + dens_func)))
        return pval


# Gaussian distribution
class Gaussian(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def has_dispersion():
        return False

    # method of moments
    @staticmethod
    def mom(x):
        raise NotImplementedError

    @staticmethod
    def loss(x_true, x_pred, **kwargs):
        x_na = tfm.is_finite(x_true)
        return tf.keras.losses.MeanSquaredError()(tf.boolean_mask(x_true,
                                                                  x_na),
                                                  tf.boolean_mask(x_pred,
                                                                  x_na))

    @classmethod
    def calc_pvalues(cls, x_true, x_pred, parallel_iterations=1,
                     dispersions=None):
        # pval calculation
        pvalues = cls.tf_get_pval(x_true, x_pred,
                                  parallel_iterations=parallel_iterations)
        return pvalues

    @classmethod
    @tf.function
    def tf_get_pval(cls, X, X_pred, parallel_iterations):
        X_cols = tf.range(tf.shape(X)[1], dtype=tf.int32)
        pval = tf.map_fn(lambda x: (cls._tf_get_pval_gaus(
                                            X=X[:, x],
                                            X_pred=X_pred[:, x])),
                         X_cols,
                         fn_output_signature=X.dtype,
                         parallel_iterations=parallel_iterations)
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


# Log-gaussian distribution
class Log_Gaussian(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def has_dispersion():
        return False

    # method of moments
    @staticmethod
    def mom(x):
        raise NotImplementedError

    @staticmethod
    def loss(x_true, x_pred, **kwargs):
        x_log = tfm.log1p(x_true)
        x_pred_log = tfm.log1p(x_pred)

        x_na = tfm.is_finite(x_log)
        return tf.keras.losses.MeanSquaredError()(tf.boolean_mask(x_log, x_na),
                                                  tf.boolean_mask(x_pred_log,
                                                                  x_na))

    @classmethod
    def calc_pvalues(cls, x_true, x_pred, parallel_iterations=1, **kwargs):
        pvalues = cls.tf_get_pval(tfm.log1p(x_true), tfm.log1p(x_pred),
                                  parallel_iterations=parallel_iterations)
        return pvalues

    @classmethod
    @tf.function
    def tf_get_pval(cls, X, X_pred, parallel_iterations):
        X_cols = tf.range(tf.shape(X)[1], dtype=tf.int32)
        pval = tf.map_fn(lambda x: (Gaussian._tf_get_pval_gaus(
                                                        X=X[:, x],
                                                        X_pred=X_pred[:, x])),
                         X_cols,
                         dtype=X.dtype,
                         parallel_iterations=parallel_iterations)
        return tf.transpose(pval)


DISTRIBUTIONS = {'gaussian': Gaussian, 'NB': NB, 'log-gaussian': Log_Gaussian}
