from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf    # 2.0.0
from tensorflow import math as tfm



# from ae_tf_functions import vst_transform

import ae_models.tf_init
import utilis.float_limits
from utilis.tf_fminbound import tf_fminbound
from distributions.tf_loss_func import tf_neg_bin_loss
from utilis.np_mom_theta import robust_mom_theta
from dataset_handling.ae_dataset import Ae_dataset
from distributions.dis_neg_bin import Dis_neg_bin
from distributions.dis_gaussian import Dis_gaussian
from distributions.transform_func import rev_transform_ae_input


class Ae_abstract(ABC):

    def __init__(self, ae_dataset):
        self.ds = ae_dataset
        self.ds.initialize_ds()

        self.loss_list = None

        ae_models.tf_init.init_tf_config(num_cpus=self.ds.xrds.attrs["num_cpus"], verbose=self.ds.xrds.attrs["verbose"])
        ae_models.tf_init.init_float_type(float_type=self.ds.xrds.attrs["float_type"])




    @property
    def ds(self):
        return self.__ds

    @ds.setter
    def ds(self, ds):
        self.__ds = ds


    @abstractmethod
    def run_fit(self):
         pass


    def run_autoencoder(self, **kwargs):
        self.run_fit(**kwargs)
        self.calc_X_pred()
        self.ds.init_pvalue_fc_z()
        self.xrds = self.ds.get_xrds()
        return self.xrds



    ##### prediction calculation steps

    def _pred_X_trans(self, H, D, b):
        y = np.matmul(H, D) # y: sample x gene
        y = y[:,0:len(b)]  # avoid cov_sample inclusion
        y_b = y + b
        y_b = utilis.float_limits.min_value_exp(y_b)
        return y_b

    def _pred_X(self, profile, H, D, b, par_sample):
        y = self._pred_X_trans(H, D, b)
        return rev_transform_ae_input(y, profile.ae_input_trans, par_sample=par_sample)


    def calc_X_pred(self):
        self.ds.X_trans_pred = self._pred_X_trans(self.ds.H, self.ds.D, self.ds.b)
        self.ds.X_pred = self._pred_X(self.ds.profile, self.ds.H, self.ds.D, self.ds.b, self.ds.par_sample)



    def get_loss(self):
        self.calc_X_pred()
        ds_dis = self.ds.profile.distribution(X=self.ds.X, X_pred=self.ds.X_pred,
                                           par=self.ds.par_meas, parallel_iterations=self.ds.parallel_iterations)
        loss = ds_dis.get_loss()
        return loss





