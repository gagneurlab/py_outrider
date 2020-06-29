import numpy as np
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


class Ae_pca(Ae_abstract):


    def __init__(self, ae_dataset):
        super().__init__(ae_dataset)

        self.ds.ae_input = self.ds.X_trans  # no covariate consideration in pca



    # @tf.function
    def run_fit(self, **kwargs):
        # time_ae_start = time.time()
        self.loss_list = Loss_list(conv_limit=0, last_iter=0)

        # print_func.print_time('pca start')
        E_pca(self.ds).fit()

        self.calc_X_pred()
        Par_meas_fminbound(self.ds).fit()

        self.get_loss()


        # self.loss_list.add_loss(self.get_loss(), step_name='pca', print_text='pca end with loss:      ')
        #
        # print_func.print_time(f'complete ae time {print_func.get_duration_sec(time.time() - time_ae_start)}')



