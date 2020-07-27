import numpy as np
import tensorflow as tf    # 2.0.0
import tensorflow_probability as tfp
from sklearn.decomposition import PCA
import time



from fit_components.fitting_models.model_fit_abstract import Model_fit_abstract
# from autoencoder_models.loss_list import Loss_list
import utilis.print_func as print_func


from fit_components.latent_space_regression.D_lbfgs_single import D_lbfgs_single
from fit_components.latent_space_regression.D_lbfgs_whole import D_lbfgs_whole
from fit_components.latent_space_fit.E_pca import E_pca
from fit_components.latent_space_fit.E_lbfgs import E_lbfgs
from fit_components.par_meas_fit.par_meas_fminbound import Par_meas_fminbound
from fit_components.par_meas_fit.par_meas_mom import Par_meas_mom



class Model_fit_lbfgs(Model_fit_abstract):


    def __init__(self, ae_dataset):
        super().__init__(ae_dataset)



    # @tf.function
    def fit(self, conv_limit=1e-5, **kwargs):

        E_pca(ds=self.ds).fit()

        Par_meas_mom(ds=self.ds).fit()
        # self.ds.print_dataset_shapes()
        # D_lbfgs_whole(ds=self.ds).fit()
        D_lbfgs_single(ds=self.ds).fit()

        # self.ds.print_dataset_shapes()

        Par_meas_fminbound(ds=self.ds).fit()
        # self.ds.print_dataset_shapes()

        ### ITERATE UNTIL CONVERGENCE
        for iter in range(self.ds.xrds.attrs["max_iter"]):
            print(f'### ITERATION {iter}')
            time_iter_start = time.time()

            E_lbfgs(ds=self.ds).fit()
            # self.ds.print_dataset_shapes()

            # print('### D_SINGLE MODEL FIT')
            D_lbfgs_single(ds=self.ds).fit()
            # D_lbfgs_whole(ds=self.ds).fit()

            Par_meas_fminbound(ds=self.ds).fit()

            print('duration loop: {}'.format(print_func.get_duration_sec(time.time() - time_iter_start)))

            ## check convergence
            if self.ds.loss_list.check_converged(conv_limit=conv_limit, verbose=self.ds.xrds.attrs["verbose"]):
                print_func.print_time(f'model converged at iteration: {iter}')
                break



# max_iter=15
# for i in 10:
    #
    # class Model_fit_lbfgs(Model_fit_abstract):
    #
    #     def __init__(self, ae_dataset):
    #         super().__init__(ae_dataset)
    #
    #
    #     @tf.function
    #     def fit(self, **kwargs):
    #
    #         E_pca(ds=self.ds).fit()
    #         Par_meas_mom(ds=self.ds).fit()
    #         D_lbfgs_whole(ds=self.ds).fit()
    #         Par_meas_fminbound(ds=self.ds).fit()
    #
    #         for iter in range(max_iter):
    #             E_lbfgs(ds=self.ds).fit()
    #             D_lbfgs_single(ds=self.ds).fit()
    #             Par_meas_fminbound(ds=self.ds).fit()
    #











