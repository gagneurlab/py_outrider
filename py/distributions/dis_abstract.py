from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import tensorflow as tf



class Dis_abstract(ABC):

    def __init__(self, y_true, y_pred, par, float_type):
        self.y_true = y_true
        self.y_pred = y_pred
        self.par = par


    @property
    def y_true(self):
        return self.__y_true

    @y_true.setter
    def y_true(self, y_true):

        if not tf.is_tensor(y_true):
            if isinstance(y_true, np.ndarray):
                y_true = tf.convert_to_tensor(y_true, dtype=self.float_type)
            else:
                y_true = tf.convert_to_tensor(y_true.values, dtype=self.float_type)

        self.__y_true= y_true


    @property
    def y_pred(self):
        return self.__y_pred

    @y_pred.setter
    def y_pred(self, y_pred):

        if not tf.is_tensor(y_pred):
            if isinstance(y_pred, np.ndarray):
                y_pred = tf.convert_to_tensor(y_pred, dtype=self.float_type)
            else:
                y_pred = tf.convert_to_tensor(y_pred.values, dtype=self.float_type)

        self.__y_pred= y_pred


    @property
    def par(self):
        return self.__par

    @par.setter
    def par(self, par):
        self.__par= par


    @property
    def float_type(self):
        return self.__float_type

    @float_type.setter
    def float_type(self, float_type):
        self.__float_type = float_type



    @abstractmethod
    def get_pval(self):
         pass

    @abstractmethod
    def get_pval_adj(self):
         pass

    @abstractmethod
    def get_loss(self):
        pass

