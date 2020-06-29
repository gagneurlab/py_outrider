from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import tensorflow as tf



class D_abstract(ABC):

    def __init__(self, ds):
        self.ds = ds


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
    def update_weights(self):
        pass







