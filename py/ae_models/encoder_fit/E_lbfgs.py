import numpy as np
import tensorflow as tf    # 2.0.0
from tensorflow import math as tfm
import tensorflow_probability as tfp
import time

from ae_models.ae_abstract import Ae_abstract
# from autoencoder_models.loss_list import Loss_list
import utilis.print_func as print_func
import utilis.float_limits
from ae_models.loss_list import Loss_list

from ae_models.encoder_fit import E_abstract


class E_lbfgs(E_abstract):

    def __init__(self, **kwargs):
        self.__init__(**kwargs)



    def run_fit(self):
        E_optim_obj = self.get_updated_E(loss_func = self.ds.profile.loss_E,
                                       E = self.ds.E, D= self.ds.D, b = self.ds.b, x = self.ds.X, X_trans = self.ds.ae_input_noise, cov_sample=self.ds.cov_sample,
                                       par_sample = self.ds.par_sample, par_meas = self.ds.par_meas, parallel_iterations=self.ds.parallel_iterations)

        E, H = self.reshape_e_to_H(E_optim_obj["E_optim"], self.ds.ae_input, self.ds.X, self.ds.D, self.ds.cov_sample)
        self.update_weights(E=E, H=H)



    def update_weights(self, E, H):
        self.ds.E = tf.convert_to_tensor(E, dtype=self.ds.X.dtype)
        self.ds.H = tf.convert_to_tensor(H, dtype=self.ds.X.dtype)





    @tf.function
    def get_updated_E(self, loss_func, E, D, b, x, X_trans, par_sample, par_meas, cov_sample, parallel_iterations=1):
        e = tf.reshape(E, shape=[tf.size(E), ])

        def lbfgs_input(e):
            loss = loss_func(e, D, b, x, X_trans, par_sample, par_meas, cov_sample)
            gradients = tf.gradients(loss, e)[0]
            return loss, tf.clip_by_value(gradients, -100., 100.)

        optim = tfp.optimizer.lbfgs_minimize(lbfgs_input, initial_position=e, tolerance=1e-8, max_iterations=100, #500, #150,
                                             num_correction_pairs=10, parallel_iterations=parallel_iterations)

        #
        # ### transform back in correct shape
        # if x.shape == X_trans.shape:  # no covariates in encoding step
        #     if cov_sample is None:
        #         E_shape = tf.shape(tf.transpose(D))    # meas x encod_dim
        #     else:
        #         E_shape = (tf.shape(D)[1], (tf.shape(D)[0] - tf.shape(cov_sample)[1]))   # meas x encod_dim
        # else:
        #     E_shape = (tf.shape(X_trans)[1], tf.shape(D)[0] - tf.shape(cov_sample)[1]) # meas+cov x encod_dim
        # out_E = tf.reshape(optim.position, E_shape)

        # if self.ds_obj.verbose:
        #     print_func.print_lbfgs_optimizer(optim)

        return {"E_optim":optim.position, "optim":optim }    # e and optimizer






