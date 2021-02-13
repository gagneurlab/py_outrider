from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import tensorflow as tf

from ..utils.np_mom_theta import robust_mom_theta
from ..utils.tf_fminbound import tf_fminbound

class Dispersions_fit(ABC):

    def __init__(self, ds):
        self.ds = ds
        self.loss_dispersions = self.ds.profile.loss_dis.tf_loss


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
    def _update_dispersions(self):
        pass



    def fit(self):
        self.run_fit()
        self.ds.calc_X_pred()
        dispersions_name = self.__class__.__name__
        self.ds.loss_list.add_loss(self.ds.get_loss(), step_name=dispersions_name, print_text=f'{dispersions_name} - loss:')


### Fit disperions with method of moments
class Dispersions_mom(Dispersions_fit):

    def __init__(self, theta_range = (1e-2, 1e3),*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta_range = theta_range

    def run_fit(self):
        if self.ds.profile.dis.__name__ == "Dis_neg_bin":
            dispersions = robust_mom_theta(self.ds.X, self.theta_range[0], self.theta_range[1])
        else:
            dispersions = np.zeros(shape=(self.ds.X.shape[1],))
        self._update_dispersions(dispersions)


    def _update_dispersions(self, dispersions):
        self.ds.dispersions =  tf.convert_to_tensor(dispersions, dtype=self.ds.X.dtype)
        
        
### Fit dispersions by optimizing loss function
class Dispersions_fminbound(Dispersions_fit):

    def __init__(self, theta_range = (1e-2, 1e3), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta_range = theta_range

    def run_fit(self):
        if self.ds.profile.dis.__name__ == "Dis_neg_bin":
            dispersions = self.update_dispersions_fmin(loss_func=self.loss_dispersions, x=self.ds.X, x_pred=self.ds.X_pred,
                                                 par_list=self.theta_range,
                                                 parallel_iterations=self.ds.parallel_iterations)
        else:
            dispersions = np.zeros(shape=(self.ds.X.shape[1],))
        self._update_dispersions(dispersions)


    def _update_dispersions(self, dispersions):
        self.ds.dispersions =  tf.convert_to_tensor(dispersions, dtype=self.ds.X.dtype)

    def update_dispersions_fmin(self, loss_func, x, x_pred, par_list, parallel_iterations=1):

            @tf.function(experimental_relax_shapes=True)
            def my_map(*args, **kwargs):
                return tf.map_fn(*args, **kwargs)

            y_true_pred_stacked = tf.transpose(tf.stack([x, x_pred], axis=1))
            cov_meas = my_map(
                lambda row: tf_fminbound(
                    lambda t: loss_func(x=row[0, :], x_pred=row[1, :], dispersions=t),
                    x1=tf.constant(par_list[0], dtype=x.dtype),
                    x2=tf.constant(par_list[1], dtype=x.dtype)),
                y_true_pred_stacked, parallel_iterations=parallel_iterations)

            return cov_meas



