import numpy as np
import tensorflow as tf    # 2.0.0
from tensorflow import math as tfm
import tensorflow_probability as tfp
from sklearn.decomposition import PCA
import time

from ae_models.ae_abstract import Ae_abstract
# from autoencoder_models.loss_list import Loss_list
import utilis.print_func as print_func
import utilis.float_limits
from ae_models.loss_list import Loss_list

from ae_models.encoder_fit import E_abstract


class E_pca(E_abstract):

    def __init__(self, **kwargs):
        self.__init__(**kwargs)



    def run_fit(self):
        pca = PCA(n_components=self.ds.xrds.attrs["encod_dim"], svd_solver='full')
        pca.fit(self.ds.ae_input)
        pca_coef = pca.components_  # encod_dim x samples
        self.update_weights(pca_coef)



    def update_weights(self, pca_coef):
        self.ds.E = tf.convert_to_tensor(np.transpose(pca_coef), dtype=self.ds.X.dtype)
        if self.ds.D is None:
            self.ds.D = tf.convert_to_tensor(pca_coef, dtype=self.ds.X.dtype)
            self.ds.b = tf.convert_to_tensor(self.ds.X_center_bias, dtype=self.ds.X.dtype)

        E, H = self.reshape_e_to_H(self.ds.E, self.ds.ae_input, self.ds.X, self.ds.D, self.ds.cov_sample)
        self.ds.H = H















