from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import tensorflow as tf



class Dis_abstract(ABC):

    def __init__(self, X_true, X_pred, par, parallel_iterations):
        self.float_type = X_pred.dtype  # keep init first
        self.X_true = X_true
        self.X_pred = X_pred
        self.par = par
        self.parallel_iterations = parallel_iterations
        self.pvalue = None


    @property
    def X_true(self):
        return self.__X_true

    @X_true.setter
    def X_true(self, X_true):
        if not tf.is_tensor(X_true):
            if isinstance(X_true, np.ndarray):
                X_true = tf.convert_to_tensor(X_true, dtype=self.float_type)
            else:
                X_true = tf.convert_to_tensor(X_true.values, dtype=self.float_type)

        self.__X_true= X_true


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

    # def _type(self):
    #     return self.__class__.__name__