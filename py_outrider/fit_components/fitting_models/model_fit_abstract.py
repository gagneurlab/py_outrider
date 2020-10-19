# from ae_tf_functions import vst_transform
import time
from abc import ABC, abstractmethod

import py_outrider.fit_components.tf_init
# from autoencoder_models.loss_list import Loss_list
import py_outrider.utils.print_func as print_func


# from ae_tf_functions import vst_transform


class Model_fit_abstract(ABC):

    def __init__(self, model_dataset):
        self.ds = model_dataset
        self.ds.initialize_ds()

        py_outrider.fit_components.tf_init.init_tf_config(num_cpus=self.ds.xrds.attrs["num_cpus"], verbose=self.ds.xrds.attrs["verbose"])
        py_outrider.fit_components.tf_init.init_float_type(float_type=self.ds.xrds.attrs["float_type"])
        py_outrider.fit_components.tf_init.init_tf_seed(seed=self.ds.xrds.attrs["seed"])



    @property
    def ds(self):
        return self.__ds

    @ds.setter
    def ds(self, ds):
        self.__ds = ds

    @abstractmethod
    def fit(self):
         pass


    def run_model_fit(self, **kwargs):
        time_ae_start = time.time()
        print_func.print_time('start model fitting')
        self.fit(**kwargs)
        print_func.print_time(f'complete model fit time {print_func.get_duration_sec(time.time() - time_ae_start)}')
        print_func.print_time(f'model_fit ended with loss: {self.ds.get_loss()}')

        self.ds.init_pvalue_fc_z()
        self.xrds = self.ds.get_xrds()
        return self.xrds










