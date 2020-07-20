
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import tensorflow as tf
from fit_components.par_meas_fit.par_meas_abstract import Par_meas_abstract
from utilis.tf_fminbound import tf_fminbound



class Par_meas_fminbound(Par_meas_abstract):


    def __init__(self, theta_range = (1e-2, 1e3), *args, **kwargs):
        super().__init__(*args, **kwargs)


        self.theta_range = theta_range


    def run_fit(self):
        if self.ds.profile.dis.__name__ == "Dis_neg_bin":
            par_meas = self.update_par_meas_fmin(loss_func=self.loss_par_meas, x=self.ds.X, x_pred=self.ds.X_pred,
                                                 par_list=self.theta_range,
                                                 parallel_iterations=self.ds.parallel_iterations)
        else:
            par_meas = np.zeros(shape=(self.ds.X.shape[1],))
        self._update_par_meas(par_meas)


    def _update_par_meas(self, par_meas):
        self.ds.par_meas =  tf.convert_to_tensor(par_meas, dtype=self.ds.X.dtype)



    def update_par_meas_fmin(self, loss_func, x, x_pred, par_list, parallel_iterations=1):

            @tf.function
            def my_map(*args, **kwargs):
                return tf.map_fn(*args, **kwargs)

            y_true_pred_stacked = tf.transpose(tf.stack([x, x_pred], axis=1))
            cov_meas = my_map(
                lambda row: tf_fminbound(
                    lambda t: loss_func(x=row[0, :], x_pred=row[1, :], par_meas=t),
                    x1=tf.constant(par_list[0], dtype=x.dtype),
                    x2=tf.constant(par_list[1], dtype=x.dtype)),
                y_true_pred_stacked, parallel_iterations=parallel_iterations)

            return cov_meas












