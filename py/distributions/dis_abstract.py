from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import tensorflow as tf



class Dis_abstract(ABC):

    def __init__(self, X_true, X_pred, par, parallel_iterations):
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
                X_true = tf.convert_to_tensor(X_true, dtype=X_true.dtype)
            else:
                X_true = tf.convert_to_tensor(X_true.values, dtype=X_true.dtype)

        self.__X_true= X_true


    @property
    def X_pred(self):
        return self.__X_pred

    @X_pred.setter
    def X_pred(self, X_pred):

        if not tf.is_tensor(X_pred):
            if isinstance(X_pred, np.ndarray):
                X_pred = tf.convert_to_tensor(X_pred, dtype=X_pred.dtype)
            else:
                X_pred = tf.convert_to_tensor(X_pred.values, dtype=X_pred.dtype)

        self.__X_pred= X_pred


    @property
    def par(self):
        return self.__par

    @par.setter
    def par(self, par):
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

