from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import tensorflow as tf



class Dis_abstract(ABC):

    def __init__(self, X, X_pred, par, parallel_iterations):
        self.float_type = X_pred.dtype  # keep init first
        self.X = X
        self.X_pred = X_pred
        self.par = par
        self.parallel_iterations = parallel_iterations
        self.pvalue = None


    @property
    def X(self):
        return self.__X

    @X.setter
    def X(self, X):
        if not tf.is_tensor(X):
            if isinstance(X, np.ndarray):
                X = tf.convert_to_tensor(X, dtype=self.float_type)
            else:
                X = tf.convert_to_tensor(X.values, dtype=self.float_type)

        self.__X= X


    @property
    def X_pred(self):
        return self.__X_pred

    @X_pred.setter
    def X_pred(self, X_pred):
        if not tf.is_tensor(X_pred):
            if isinstance(X_pred, np.ndarray):
                X_pred = tf.convert_to_tensor(X_pred, dtype=self.float_type)
            else:
                X_pred = tf.convert_to_tensor(X_pred.values, dtype=self.float_type)

        self.__X_pred= X_pred

    @property
    @abstractmethod
    def dis_name(self):
        pass

    @property
    def par(self):
        return self.__par

    @par.setter
    def par(self, par):
        if not tf.is_tensor(par):
            if isinstance(par, np.ndarray):
                par = tf.convert_to_tensor(par, dtype=self.float_type)
            else:
                par = tf.convert_to_tensor(par.values, dtype=self.float_type)
        self.__par= par


    @property
    def parallel_iterations(self):
        return self.__parallel_iterations

    @parallel_iterations.setter
    def parallel_iterations(self, parallel_iterations):
        self.__parallel_iterations= parallel_iterations

    @property
    def pvalue(self):
        return self.__pvalue

    @pvalue.setter
    def pvalue(self, pvalue):
        self.__pvalue= pvalue



    @abstractmethod
    def get_pvalue(self):
         pass

    @abstractmethod
    def get_pvalue_adj(self):
         pass

    @abstractmethod
    def get_loss(self):
        pass

    # @abstractmethod
    # def inject_outlier(self):
    #     pass

    @abstractmethod
    def get_random_values(self, par):
        pass


    def get_injected_outlier(x, x_norm, norm_name, inj_freq, inj_mean, inj_sd, noise_factor, log, **kwargs):
        outlier_mask = np.random.choice([0, -1, 1], size=x_norm.shape, p=[1 - inj_freq, inj_freq / 2, inj_freq / 2])

        if log:
            log_mean = np.log(inj_mean) if inj_mean != 0 else 0
            z_score = np.random.lognormal(mean=log_mean, sigma=np.log(inj_sd), size=x_norm.shape)
        else:
            z_score = np.random.normal(loc=inj_mean, scale=inj_sd, size=x_norm.shape)
        inj_values = np.abs(z_score) * noise_factor * np.nanstd(x_norm, ddof=1, axis=0)
        x_norm_outlier = inj_values * outlier_mask + x_norm

        ### avoid inj outlier to be too strong
        max_outlier_value = np.nanmin([10 * np.nanmax(x), np.iinfo("int64").max])
        cond_value_too_big = rev_normalize_ae_input(x_norm_outlier, norm_name, **kwargs) > max_outlier_value
        x_norm_outlier[cond_value_too_big] = x_norm[cond_value_too_big]
        outlier_mask[cond_value_too_big] = 0

        x_outlier = rev_normalize_ae_input(x_norm_outlier, norm_name, **kwargs)
        return {"X_norm_outlier": x_norm_outlier, "X_outlier": x_outlier, "X_is_outlier": outlier_mask}















