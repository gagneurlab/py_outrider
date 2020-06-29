from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import tensorflow as tf
from ae_models.par_meas_fit.par_meas_abstract import Par_meas_abstract
from utilis.tf_fminbound import tf_fminbound

class Par_meas_fminbound(Par_meas_abstract):

    def __init__(self, **kwargs):
        self.__init__(**kwargs)


    def fit(self):
        pass

    def update_par_meas(self):
        self.ds.par_meas = 5






from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import tensorflow as tf
from ae_models.par_meas_fit.par_meas_abstract import Par_meas_abstract
from utilis.tf_fminbound import tf_fminbound
from distributions.tf_loss_func import tf_neg_bin_loss

class Par_meas_fminbound(Par_meas_abstract):

    def __init__(self, theta_range = (1e-2, 1e3), **kwargs):
        self.__init__(**kwargs)
        self.theta_range = theta_range


    def fit(self):
        if self.ds.profile.distribution.dis_name == "Dis_neg_bin":
            par_meas = self.update_par_meas_fmin(tf_neg_bin_loss, X=self.ds.X, X_pred=self.ds.X_pred,
                                                 par_list=self.theta_range,
                                                 parallel_iterations=self.ds.parallel_iterations)
        else:
            par_meas = np.zeros(shape=(self.ds.X.shape[1],))
        self._update_par_meas(par_meas)


    def _update_par_meas(self, par_meas):
        self.ds.par_meas =  tf.convert_to_tensor(par_meas, dtype=self.ds.X.dtype)













