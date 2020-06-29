from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import tensorflow as tf



class Par_meas_abstract(ABC):

    def __init__(self, ds):
        self.ds = ds


    @property
    def ds(self):
        return self.__ds

    @ds.setter
    def ds(self, ds):
        self.__ds = ds

    @property
    @abstractmethod
    def par_meas_name(self):
        pass


    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def _update_par_meas(self):
        pass



    def run_fit(self):
        self.fit()
        self.ds.calc_X_pred()
        self.ds.loss_list.add_loss(self.ds.get_loss(), step_name=self.par_meas_name, print_text=f'{self.par_meas_name} - loss:')









