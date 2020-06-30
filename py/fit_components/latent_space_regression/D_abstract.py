from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import tensorflow as tf



class D_abstract(ABC):

    def __init__(self, ds):
        self.ds = ds
        self.loss_D = self.ds.profile.loss_dis.tf_loss_D

    @property
    def ds(self):
        return self.__ds

    @ds.setter
    def ds(self, ds):
        self.__ds = ds


    @property
    @abstractmethod
    def D_name(self):
        pass



    @property
    def loss_D(self):
        return self.__loss_D

    @loss_D.setter
    def loss_D(self, loss_D):
        self.__loss_D = loss_D



    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def _update_weights(self):
        pass


    def run_fit(self):
        self.fit()
        self.ds.calc_X_pred()
        self.ds.loss_list.add_loss(self.ds.get_loss(), step_name=self.D_name, print_text=f'{self.D_name} - loss:')






