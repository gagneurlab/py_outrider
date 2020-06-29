from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from dataset_handling.data_transform.transform_func import rev_transform_ae_input




class Trans_abstract(ABC):


    @property
    @abstractmethod
    def trans_name(self):
        pass

    @staticmethod
    @abstractmethod
    def get_transformed_xrds(self):
        pass

    @staticmethod
    @abstractmethod
    def rev_transform(self):
        pass

    @staticmethod
    def transform_xrds(xrds):
        X_trans = Trans_abstract.get_transformed_xrds(xrds)

        ### center to zero
        X_bias = np.mean(X_trans, axis=0)
        X_trans_cen = X_trans - X_bias

        xrds["X_trans"] = (('sample', 'meas'), X_trans_cen)
        xrds["X_center_bias"] = (('meas'), X_bias)





