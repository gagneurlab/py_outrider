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


    def __init__(self, xrds):
        super().__init__(xrds = xrds)

        ### covariate consideration
        if self.ae_ds.cov_sample is not None:
            self.ae_input = np.concatenate([self.ae_ds.X_norm , self.ae_ds.cov_sample], axis=1)
            print("used cov_sample")
        else:
            self.ae_input = self.ae_ds.X_norm



    # @tf.function
    def run_autoencoder(self,  theta_range=(1e-2, 1e3), **kwargs):
        time_ae_start = time.time()
        self.loss_list = Loss_list(conv_limit=0, last_iter=0)




        ### initialize tensor weights with pca
        print_func.print_time('pca start')
        pca = PCA(n_components=self.xrds.attrs["encod_dim"], svd_solver='full')
        pca.fit(self.ae_input)
        pca_coef = pca.components_  # (10,200)

        self.E = np.transpose(pca_coef)
        self.D = pca_coef
        self.b = self.ae_ds.X_center_bias

        self.ae_ds.X_true_pred = self.get_X_true_pred()

        self.par_meas = self.get_updated_par_meas(self.ae_ds.X_true, self.ae_ds.X_true_pred,
                                                  {'init_step': False, 'theta_range': theta_range},
                                                  parallel_iterations=self.parallel_iterations)
        print('loss print')
        print(self.get_loss())
        # self.loss_list.add_loss(self.get_loss(), step_name='pca', print_text='pca end with loss:      ')

        print_func.print_time(f'complete ae time {print_func.get_duration_sec(time.time() - time_ae_start)}')



    # def return_xrds

    # self.X_norm_pred = self.get_X_norm_pred(self.ae_input, self.E, self.D, self.b)
    # self.X_pred = self.get_X_pred(self.ae_input, self.E, self.D, self.b)

    #
    #
    #
    # def get_encoder(self):
    #     return self.E.numpy()
    #
    # def get_decoder(self):
    #     return self.D.numpy(), self.b.numpy()
    #
    # def get_hidden_space(self):
    #     H = tf.matmul(self.x_norm, self.E)
    #     return H.numpy()
    #
    #
    # def get_loss_genes(self):
    #     y_pred = self.get_pred_y()
    #     ds_dis = self.ds_obj.get_distribution()(y_true=self.x, y_pred=y_pred,
    #                                           par=self.cov_meas, float_type=self.ds_obj.float_type)
    #     loss = ds_dis.get_loss_genes()
    #     return loss
    #
    #
    #



