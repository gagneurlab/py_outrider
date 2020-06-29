from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf    # 2.0.0

# from ae_tf_functions import vst_transform
import numpy as np
import tensorflow as tf    # 2.0.0
import tensorflow_probability as tfp
from sklearn.decomposition import PCA
import time

from ae_models.fitting_models.model_fit_abstract import Ae_abstract
# from autoencoder_models.loss_list import Loss_list
import utilis.print_func as print_func
from ae_models.loss_list import Loss_list

import ae_models.tf_init
import utilis.float_limits
from utilis.tf_fminbound import tf_fminbound
from distributions.tf_loss_func import tf_neg_bin_loss
from utilis.np_mom_theta import robust_mom_theta
from dataset_handling.data_transform.transform_func import rev_transform_ae_input

# from ae_tf_functions import vst_transform

import ae_models.tf_init


class Ae_abstract(ABC):

    def __init__(self, ae_dataset):
        self.ds = ae_dataset
        self.ds.initialize_ds()

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


    def run_model_fit(self, **kwargs):
        time_ae_start = time.time()
        self.run_fit(**kwargs)
        print_func.print_time(f'complete model fit time {print_func.get_duration_sec(time.time() - time_ae_start)}')

        self.ds.init_pvalue_fc_z()
        self.xrds = self.ds.get_xrds()
        return self.xrds










