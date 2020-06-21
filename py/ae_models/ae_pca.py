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





class Ae_pca(Ae_abstract):


    def __init__(self, ae_dataset):
        super().__init__(ae_dataset)


        self.ds.ae_input = self.ds.X_norm  # no covariate consideration in pca



    # @tf.function
    def run_fitting(self, theta_range=(1e-2, 1e3), **kwargs):
        time_ae_start = time.time()
        self.loss_list = Loss_list(conv_limit=0, last_iter=0)


        ### initialize tensor weights with pca
        print_func.print_time('pca start')
        pca = PCA(n_components=self.ds.xrds.attrs["encod_dim"], svd_solver='full')
        print(self.ds.ae_input)
        pca.fit(self.ds.ae_input)
        pca_coef = pca.components_  # (10,200)

        self.ds.E = np.transpose(pca_coef)
        self.ds.D = pca_coef
        self.ds.b = self.ds.X_center_bias

        self.calc_X_pred()
        self.ds.par_meas = self.get_updated_par_meas(self.ds.profile, self.ds.X_true, self.ds.X_true_pred,
                                                  {'init_step': False, 'theta_range': theta_range},
                                                  parallel_iterations=self.ds.parallel_iterations)

        self.loss_list.add_loss(self.get_loss(), step_name='pca', print_text='pca end with loss:      ')

        print_func.print_time(f'complete ae time {print_func.get_duration_sec(time.time() - time_ae_start)}')




