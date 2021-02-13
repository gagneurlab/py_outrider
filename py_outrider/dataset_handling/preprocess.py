from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

from .input_transform import Trans_sf

class Prepro_abstract(ABC):

    @staticmethod
    @abstractmethod
    def get_prepro_x(xrds):
        pass

    @classmethod
    def prepro_xrds(cls, xrds):
        xrds["X_raw"] = (('sample', 'meas'), xrds["X"])

        X = cls.get_prepro_x(xrds)

        xrds["X"] = (('sample', 'meas'), X)



class Prepro_none(Prepro_abstract):

    @staticmethod
    def get_prepro_x(xrds):
        return xrds["X"]



class Prepro_sf_log(Prepro_abstract):

    @staticmethod
    def get_prepro_x(xrds):
        counts = xrds["X"].values
        sf = Trans_sf.calc_size_factor(counts)
        return np.log((counts + 1) /  np.expand_dims(sf,1))



