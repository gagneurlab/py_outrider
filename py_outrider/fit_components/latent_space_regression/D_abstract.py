from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import tensorflow as tf



class D_abstract(ABC):

    def __init__(self, ds, parallelize=False):
        self.ds = ds
        self.parallelize = parallelize
        if parallelize is True:
            self.loss_D = self.ds.profile.loss_dis.tf_loss_D_single
        else:
            self.loss_D = self.ds.profile.loss_dis.tf_loss_D

    @property
    def ds(self):
        return self.__ds

    @ds.setter
    def ds(self, ds):
        self.__ds = ds




    @property
    def loss_D(self):
        return self.__loss_D

    @loss_D.setter
    def loss_D(self, loss_D):
        self.__loss_D = loss_D



    @abstractmethod
    def run_fit(self):
        pass

    @abstractmethod
    def _update_weights(self):
        pass


    def fit(self):
        self.run_fit()
        self.ds.calc_X_pred()
        D_name =  self.__class__.__name__
        self.ds.loss_list.add_loss(self.ds.get_loss(), step_name= D_name, print_text=f'{D_name} - loss:')






