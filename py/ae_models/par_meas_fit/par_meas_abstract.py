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


    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def _update_par_meas(self):
        pass











