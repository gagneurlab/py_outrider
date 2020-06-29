import numpy as np
import tensorflow as tf
from ae_models.par_meas_fit.par_meas_abstract import Par_meas_abstract
from utilis.np_mom_theta import robust_mom_theta

class Par_meas_mom(Par_meas_abstract):

    def __init__(self, theta_range = (1e-2, 1e3), **kwargs):
        self.__init__(**kwargs)
        self.theta_range = theta_range


    def run_fit(self):
        if self.ds.profile.distribution.dis_name == "Dis_neg_bin":
            par_meas = robust_mom_theta(self.ds.X, self.theta_range[0], self.theta_range[1])
        else:
            par_meas = np.zeros(shape=(self.ds.X.shape[1],))
        self.update_par_meas(par_meas)


    def update_par_meas(self, par_meas):
        self.ds.par_meas =  tf.convert_to_tensor(par_meas, dtype=self.ds.X.dtype)







