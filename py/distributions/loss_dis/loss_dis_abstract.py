from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import tensorflow as tf
from distributions.transform_func import rev_transform_ae_input




class Loss_dis_abstract(ABC):



    @abstractmethod
    def tf_loss(self):
         pass

    @abstractmethod
    def tf_loss_D(self):
         pass

    @abstractmethod
    def tf_loss_D_single(self):
         pass

    @abstractmethod
    def tf_loss_E(self):
        pass










