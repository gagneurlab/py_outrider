from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf




class Trans_abstract(ABC):


    @staticmethod
    @abstractmethod
    def get_transformed_xrds(self):
        pass

    @staticmethod
    @abstractmethod
    def rev_transform(self):
        pass

    @staticmethod
    @abstractmethod
    def get_logfc(X_trans, X_trans_pred, **kwargs):
        pass


    @classmethod
    def transform_xrds(cls, xrds):
        X_trans = cls.get_transformed_xrds(xrds)

        ### center to zero
        X_bias = np.nanmean(X_trans, axis=0)
        X_trans_cen = X_trans - X_bias

        xrds["X_trans"] = (('sample', 'meas'), X_trans_cen)
        xrds["X_center_bias"] = (('meas'), X_bias)





