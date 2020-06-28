from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import tensorflow as tf



class E_abstract(ABC):

    def __init__(self, ae_ds):
        self.ae_ds = ae_ds


    @property
    def ae_ds(self):
        return self.__ae_ds

    @ae_ds.setter
    def ae_ds(self, ae_ds):
        self.__ae_ds = ae_ds


    @abstractmethod
    def run_fit(self):
        pass

    @abstractmethod
    def update_weights(self):
        pass







