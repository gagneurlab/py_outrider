import numpy as np
import tensorflow as tf    # 2.0.0
import tensorflow_probability as tfp
from sklearn.decomposition import PCA
import time

from ae_models.fitting_models.ae_abstract import Ae_abstract
# from autoencoder_models.loss_list import Loss_list
import utilis.print_func as print_func
from ae_models.loss_list import Loss_list


from ae_models.decoder_fit.D_lbfgs_single import D_lbfgs_single
from ae_models.encoder_fit.E_pca import E_pca
from ae_models.encoder_fit.E_lbfgs import E_lbfgs
from ae_models.par_meas_fit.par_meas_fminbound import Par_meas_fminbound
from ae_models.par_meas_fit.par_meas_mom import Par_meas_mom



class Ae_lbfgs(Ae_abstract):


    def __init__(self, ae_dataset):
        super().__init__(ae_dataset)

        ### covariate consideration
        if self.ds.cov_sample is not None:
            self.ds.ae_input = np.concatenate([self.ds.X_trans , self.ds.cov_sample], axis=1)
            self.ds.ae_input_noise = np.concatenate([self.ds.X_trans_noise, self.ds.cov_sample], axis=1)
        else:
            self.ds.ae_input = self.ds.X_trans
            self.ds.ae_input_noise = self.ds.X_trans_noise



    # @tf.function
    def run_fit(self, convergence=1e-5, **kwargs):
        time_ae_start = time.time()
        self.loss_list = Loss_list(conv_limit=convergence, last_iter=3)


        ### initialize tensor weights with pca
        print_func.print_time('pca start')
        E_pca(self.ds).fit()


        # print(f"ae_input: {self.ds.ae_input.shape}")
        # print(f"E: {self.ds.E.shape}")
        # print(f"D: {self.ds.D.shape}")
        # print(f"b: {self.ds.b.shape}")
        # print(f"X_trans: {self.ds.ae_input.shape}")
        # print(f"X: {self.ds.X.shape}")
        # if self.ds.cov_sample is not None:
        #     print(f"cov_sample: {self.ds.cov_sample.shape}")



        self.calc_X_pred()
        # print_func.print_time('starting to compute the initial values of par_meas')

        Par_meas_mom(self.ds).fit()

        # self.loss_list.add_loss(self.get_loss(), step_name='pca', print_text='pca end with loss:      ')

        # print_func.print_time('starting the first fit of the decoder')
        D_lbfgs_single(self.ds).fit()

        # print_func.print_time('Starting the first fit of par_meas')

        Par_meas_fminbound(self.ds).fit()

        ### ITERATE UNTIL CONVERGENCE
        for iter in range(self.ds.xrds.attrs["max_iter"]):
            print(f'### ITERATION {iter}')
            time_iter_start = time.time()

            E_lbfgs(self.ds).fit()

            D_lbfgs_single(self.ds).fit()

            Par_meas_fminbound(self.ds).fit()

            print('duration loop: {}'.format(print_func.get_duration_sec(time.time() - time_iter_start)))

            # ## check convergence
            # if self.loss_list.check_converged(verbose=self.ds.xrds.attrs["verbose"]):
            #     print_func.print_time(f'ae converged with loss: {self.get_loss()}')
            #     break


        print_func.print_time(f'complete ae time {print_func.get_duration_sec(time.time() - time_ae_start)}')





