from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

from py_outrider.utils.other_func import np_set_seed



class Dis_abstract(ABC):

    def __init__(self, X, X_pred, par_meas, parallel_iterations, **kwargs):
        self.float_type = X_pred.dtype  # keep init first
        self.X = X
        self.X_pred = X_pred
        self.par_meas = par_meas
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
    def par_meas(self):
        return self.__par_meas

    @par_meas.setter
    def par_meas(self, par_meas):
        if par_meas is not None and not tf.is_tensor(par_meas):
            if isinstance(par_meas, np.ndarray):
                par_meas = tf.convert_to_tensor(par_meas, dtype=self.float_type)
            else:
                par_meas = tf.convert_to_tensor(par_meas.values, dtype=self.float_type)
        self.__par_meas= par_meas


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



    @staticmethod
    @abstractmethod
    def _get_random_values(inj_mean, inj_sd, size):
        pass


    @classmethod
    def get_random_values(cls, inj_mean, inj_sd, size, seed=None):
        return np_set_seed(cls._get_random_values, seed)(inj_mean, inj_sd, size)



    @classmethod
    def get_injected_outlier(cls, X, X_trans, X_center_bias, inj_freq, inj_mean, inj_sd, noise_factor, data_trans, seed, **kwargs):
        outlier_mask = np_set_seed(np.random.choice, seed)([0., -1., 1.], size=X_trans.shape, p=[1 - inj_freq, inj_freq / 2, inj_freq / 2])

        z_score = cls.get_random_values(inj_mean = inj_mean, inj_sd=inj_sd, size=X_trans.shape, seed=seed)

        inj_values = np.abs(z_score) * noise_factor * np.nanstd(X_trans, ddof=1, axis=0)
        X_trans_outlier = inj_values * outlier_mask + X_trans

        ### avoid inj outlier to be too strong
        max_outlier_value = np.nanmin([10 * np.nanmax(X), np.iinfo("int64").max])
        cond_value_too_big = data_trans.rev_transform(X_trans_outlier, **kwargs) > max_outlier_value
        X_trans_outlier[cond_value_too_big] = X_trans[cond_value_too_big]
        outlier_mask[cond_value_too_big] = 0
        outlier_mask[~np.isfinite(X)] = np.nan

        X_outlier = data_trans.rev_transform(X_trans_outlier + X_center_bias, **kwargs)
        return {"X_trans_outlier": X_trans_outlier, "X_outlier": X_outlier, "X_is_outlier": outlier_mask}















