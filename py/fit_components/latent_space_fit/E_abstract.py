from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import tensorflow as tf

import utilis.tf_helper_func as tfh



class E_abstract(ABC):

    def __init__(self, ds):
        self.ds = ds
        self.loss_E = self.ds.profile.loss_dis.tf_loss_E


    @property
    def ds(self):
        return self.__ds

    @ds.setter
    def ds(self, ds):
        self.__ds = ds

    @property
    @abstractmethod
    def E_name(self):
        pass


    @property
    def loss_E(self):
        return self.__loss_E

    @loss_E.setter
    def loss_E(self, loss_E):
        self.__loss_E = loss_E


    @abstractmethod
    def run_fit(self):
        pass

    @abstractmethod
    def _update_weights(self):
        pass


    def fit(self):
        self.run_fit()
        self.ds.calc_X_pred()
        self.ds.loss_list.add_loss(self.ds.get_loss(), step_name=self.E_name, print_text=f'{self.E_name} - loss:')



    ### depending on covariates, transform e weights to according matrix
    # @tf.function
    @staticmethod
    def reshape_e_to_H(e, fit_input, X, D, cov_sample):
        if X.shape == fit_input.shape:   # no covariates in encoding step
            if cov_sample is None:
                E_shape = tf.shape(tf.transpose(D))
            else:
                E_shape = (tf.shape(D)[1], (tf.shape(D)[0] - tf.shape(cov_sample)[1]))
            E = tf.reshape(e, E_shape)
            H = tfh.tf_nan_matmul(fit_input, E)
            # H = tf.concat([tf.matmul(fit_input, E), cov_sample], axis=1)  # uncomment if ae_bfgs_cov1
        else:
            E_shape = (tf.shape(fit_input)[1], tf.shape(D)[0] - tf.shape(cov_sample)[1])
            E = tf.reshape(e, E_shape ) # sample+cov x encod_dim
            H = tf.concat([tfh.tf_nan_matmul(fit_input, E), cov_sample], axis=1)
        return E, H







