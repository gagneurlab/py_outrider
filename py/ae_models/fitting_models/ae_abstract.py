from abc import ABC, abstractmethod

# from ae_tf_functions import vst_transform

import ae_models.tf_init


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









