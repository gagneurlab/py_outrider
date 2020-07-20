import numpy as np
import tensorflow as tf
from fit_components.par_meas_fit.par_meas_abstract import Par_meas_abstract
from utilis.np_mom_theta import robust_mom_theta




class Par_meas_mom(Par_meas_abstract):



    def __init__(self, theta_range = (1e-2, 1e3),*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta_range = theta_range


    def run_fit(self):
        if self.ds.profile.dis.__name__ == "Dis_neg_bin":
            par_meas = robust_mom_theta(self.ds.X, self.theta_range[0], self.theta_range[1])
        else:
            par_meas = np.zeros(shape=(self.ds.X.shape[1],))
        self._update_par_meas(par_meas)


    def _update_par_meas(self, par_meas):
        self.ds.par_meas =  tf.convert_to_tensor(par_meas, dtype=self.ds.X.dtype)







