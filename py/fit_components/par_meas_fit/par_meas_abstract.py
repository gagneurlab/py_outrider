from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import tensorflow as tf



class Par_meas_abstract(ABC):

    def __init__(self, ds):
        self.ds = ds
        self.loss_par_meas = self.ds.profile.loss_dis.tf_loss


    @property
    def ds(self):
        return self.__ds

    @ds.setter
    def ds(self, ds):
        self.__ds = ds



    @abstractmethod
    def run_fit(self):
        pass

    @abstractmethod
    def _update_par_meas(self):
        pass



    def fit(self):
        self.run_fit()
        self.ds.calc_X_pred()
        par_meas_name = self.__class__.__name__
        self.ds.loss_list.add_loss(self.ds.get_loss(), step_name=par_meas_name, print_text=f'{par_meas_name} - loss:')









