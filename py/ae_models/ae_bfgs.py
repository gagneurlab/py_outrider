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

import distributions




class Ae_bfgs(Ae_abstract):


    def __init__(self, ae_dataset):
        super().__init__(ae_dataset)

        ### covariate consideration
        if self.ds.cov_sample is not None:
            self.ds.ae_input = np.concatenate([self.ds.X_norm , self.ds.cov_sample], axis=1)
        else:
            self.ds.ae_input = self.ds.X_norm



    # @tf.function
    def run_fitting(self,  theta_range=(1e-2, 1e3), convergence=1e-5, **kwargs):
        time_ae_start = time.time()
        self.loss_list = Loss_list(conv_limit=convergence, last_iter=3)


        ### initialize tensor weights with pca
        print_func.print_time('pca start')
        pca = PCA(n_components=self.ds.xrds.attrs["encod_dim"], svd_solver='full')
        pca.fit(self.ds.ae_input)
        pca_coef = pca.components_  # (10,200)

        self.ds.E = tf.convert_to_tensor(np.transpose(pca_coef), dtype=self.ds.ae_input.dtype)

        ### find D space with covariates - remove at col pos and add at row pos
        if self.ds.cov_sample is not None:
            pca_coef_D = pca_coef[:,:-self.ds.cov_sample.shape[1]]
            cov_init_weights = np.zeros(shape=(self.ds.cov_sample.shape[1], pca_coef_D.shape[1]))
            pca_coef_D = np.concatenate([pca_coef_D, cov_init_weights], axis=0)
        else:
            pca_coef_D = pca_coef

        self.ds.D = tf.convert_to_tensor(pca_coef_D, dtype=self.ds.ae_input.dtype)
        self.ds.b = self.ds.X_center_bias


        print(f"ae_input: {self.ds.ae_input.shape}")
        print(f"E: {self.ds.E.shape}")
        print(f"D: {self.ds.D.shape}")
        print(f"b: {self.ds.b.shape}")
        print(f"X_norm: {self.ds.ae_input.shape}")
        print(f"X: {self.ds.X.shape}")
        if self.ds.cov_sample is not None:
            print(f"cov_sample: {self.ds.cov_sample.shape}")


        self.calc_X_pred()
        print_func.print_time('starting to compute the initial values of par_meas')
        self.ds.par_meas = self.get_updated_par_meas(self.ds.profile, self.ds.X_true, self.ds.X_true_pred,
                                                  {'init_step': True, 'theta_range': theta_range},
                                                  parallel_iterations=self.ds.parallel_iterations)

        self.loss_list.add_loss(self.get_loss(), step_name='pca', print_text='pca end with loss:      ')

        print_func.print_time('starting the first fit of the decoder')
        self.update_D()

        print_func.print_time('Starting the first fit of par_meas')
        self.update_par_meas({'init_step': False, 'theta_range': theta_range})

        ### ITERATE UNTIL CONVERGENCE
        for iter in range(self.ds.xrds.attrs["max_iter"]):
            print(f'### ITERATION {iter}')
            time_iter_start = time.time()

            self.update_E()

            self.update_D()

            self.update_par_meas({'init_step': False, 'theta_range': theta_range})

            print('duration loop: {}'.format(print_func.get_duration_sec(time.time() - time_iter_start)))


            ## check convergence
            if self.loss_list.check_converged(verbose=self.ds.xrds.attrs["verbose"]):
                print_func.print_time(f'ae converged with loss: {self.get_loss()}')
                break


        print_func.print_time(f'complete ae time {print_func.get_duration_sec(time.time() - time_ae_start)}')







    ##### UPDATE ENCODER STEPS #####
    def update_E(self):
        updated_E = self.get_updated_E(loss_func = self.ds.profile.loss_E,
                                       E = self.ds.E, D= self.ds.D, b = self.ds.b, x = self.ds.X_true, x_norm = self.ds.ae_input, cov_sample=self.ds.cov_sample,
                                       par_sample = self.ds.par_sample, par_meas = self.ds.par_meas, parallel_iterations=self.ds.parallel_iterations)

        if (self.ds.E == updated_E[0]).numpy().all():
            print('### error: did not update E')

        self.ds.E = updated_E[0]
        self.loss_list.add_loss(self.get_loss(), step_name='E', print_text='E loss:     ')

        # if self.ds_obj.verbose:
        #     self.loss_list.check_converged(True)




    @tf.function
    def get_updated_E(self, loss_func, E, D, b, x, x_norm, par_sample, par_meas, cov_sample, parallel_iterations=1):
        e = tf.reshape(E, shape=[tf.size(E), ])

        def lbfgs_input(e):
            loss = loss_func(e, D, b, x, x_norm, par_sample, par_meas, cov_sample)
            gradients = tf.gradients(loss, e)[0]
            return loss, tf.clip_by_value(gradients, -100., 100.)

        optim = tfp.optimizer.lbfgs_minimize(lbfgs_input, initial_position=e, tolerance=1e-8, max_iterations=100, #500, #150,
                                             num_correction_pairs=10, parallel_iterations=parallel_iterations)

        ### transform back in correct shape
        if x.shape == x_norm.shape:  # no covariates in encoding step
            if cov_sample is None:
                E_shape = tf.shape(tf.transpose(D))    # meas x encod_dim
            else:
                E_shape = (tf.shape(D)[1], (tf.shape(D)[0] - tf.shape(cov_sample)[1]))   # meas x encod_dim
        else:
            E_shape = (tf.shape(x_norm)[1], tf.shape(D)[0] - tf.shape(cov_sample)[1]) # meas+cov x encod_dim
        out_E = tf.reshape(optim.position, E_shape)


        # if self.ds_obj.verbose:
        #     print_func.print_lbfgs_optimizer(optim)

        return out_E, optim






    ##### UPDATE DECODER STEPS #####
    def update_D(self):
        updated_D = self.get_updated_D(loss_func = self.ds.profile.loss_D,
                                       E = self.ds.E, x_norm = self.ds.ae_input, x = self.ds.X_true, b = self.ds.b, D= self.ds.D,  cov_sample=self.ds.cov_sample,
                                       par_sample = self.ds.par_sample, par_meas = self.ds.par_meas, parallel_iterations=self.ds.parallel_iterations)

        if (self.ds.D == updated_D[1]).numpy().all():
            print('### error: did not update D')

        self.ds.b = updated_D[0]
        self.ds.D = updated_D[1]
        self.loss_list.add_loss(self.get_loss(), step_name='D', print_text='D loss:     ')

        # if self.ds_obj.verbose:
        #     self.loss_list.check_converged(True)
            # self.loss_list_genes.check_converged(True)  # TODO uncomment




    @tf.function
    def get_updated_D(self, loss_func, E, x_norm, x, b, D, cov_sample, par_sample, par_meas, parallel_iterations=1):
        if cov_sample is None:
            H = tf.matmul(x_norm, E)
        else:
            H = tf.concat([tf.matmul(x_norm, E), cov_sample], axis=1)

        meas_cols = tf.range(tf.shape(D)[1])
        map_D = tf.map_fn(lambda i: (self.single_fit_D(loss_func, H, x[:, i], b[i], D[:, i], par_sample, par_meas[i])),
                          meas_cols,
                          dtype=x.dtype,
                          parallel_iterations=parallel_iterations)

        return map_D[:, 0], tf.transpose(map_D[:, 1:])  # returns b and D


    ### single vector/gene fit if tf.map_fn is used
    @tf.function
    def single_fit_D(self, loss_func, H, x_i, b_i, D_i, par_sample, par_meas_i):
        b_and_D = tf.concat([tf.expand_dims(b_i, 0), D_i], axis=0)

        def lbfgs_input(b_and_D):
            loss = loss_func(H, x_i, b_and_D, par_sample, par_meas_i)
            gradients = tf.gradients(loss, b_and_D)[0]
            return loss, tf.clip_by_value(gradients, -100., 100.)

        optim = tfp.optimizer.lbfgs_minimize(lbfgs_input, initial_position=b_and_D, tolerance=1e-6,
                                             max_iterations=50)

        return optim.position  # b(200) and D(200x10) -> needs t()






    ##### UPDATE PAR_MEAS STEPS #####
    def update_par_meas(self, par):
        if self.ds.par_meas is not None:
            par_meas_temp = self.ds.par_meas
            self.calc_X_pred()
            self.ds.par_meas = self.get_updated_par_meas(self.ds.profile, self.ds.X_true, self.ds.X_true_pred,
                                      par,  parallel_iterations=self.ds.parallel_iterations)
            self.loss_list.add_loss(self.get_loss(), step_name='theta', print_text='theta loss: ')

            if (par_meas_temp == self.ds.par_meas).numpy().all():
                print('### error: did not update par_meas')




    ### hidden space with inclusion
    def _pred_H(self, ae_input, E, cov_sample):
        H = np.matmul(ae_input, E)
        if cov_sample is not None:
            H = np.concatenate([H, cov_sample], axis=1)
        return H


