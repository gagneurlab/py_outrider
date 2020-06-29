from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import tensorflow as tf



class E_abstract(ABC):

    def __init__(self, ds):
        self.ds = ds


    @property
    def ds(self):
        return self.__ds

    @ds.setter
    def ds(self, ds):
        self.__ds = ds


    @abstractmethod
    def run_fit(self):
        pass

    @abstractmethod
    def update_weights(self):
        pass



    ### depending on covariates, transform e weights to according matrix
    @tf.function
    def reshape_e_to_H(self, e, ae_input, X, D, cov_sample):
        if X.shape == ae_input.shape:   # no covariates in encoding step
            if cov_sample is None:
                E_shape = tf.shape(tf.transpose(D))
            else:
                E_shape = (tf.shape(D)[1], (tf.shape(D)[0] - tf.shape(cov_sample)[1]))
            E = tf.reshape(e, E_shape)
            H = tf.matmul(ae_input, E)  #
            # H = tf.concat([tf.matmul(ae_input, E), cov_sample], axis=1)  # uncomment if ae_bfgs_cov1
        else:
            E_shape = (tf.shape(ae_input)[1], tf.shape(D)[0] - tf.shape(cov_sample)[1])
            E = tf.reshape(e, E_shape ) # sample+cov x encod_dim
            H = tf.concat([tf.matmul(ae_input, E), cov_sample], axis=1)
        return E, H









