from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf




class Prepro_abstract(ABC):


    @property
    @abstractmethod
    def prepro_name(self):
        pass


    @staticmethod
    @abstractmethod
    def get_prepro_x(xrds):
        pass



    @classmethod
    def prepro_xrds(cls, xrds):
        xrds["X_raw"] = (('sample', 'meas'), xrds["X"])

        X = cls.get_prepro_x(xrds)

        xrds["X"] = (('sample', 'meas'), X)





