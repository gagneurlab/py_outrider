from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf    # 2.0.0

# from ae_tf_functions import vst_transform

import ae_models.tf_init
import utilis.float_limits
from utilis.tf_fminbound import tf_fminbound
from distributions.tf_loss_func import tf_neg_bin_loss
from utilis.np_mom_theta import robust_mom_theta
from distributions.transform_func import rev_transform_ae_input


class Ae_abstract(ABC):

    def __init__(self, ae_dataset):
        self.ds = ae_dataset
        self.ds.initialize_ds()

        self.loss_list = None

        ae_models.tf_init.init_tf_config(num_cpus=self.ds.xrds.attrs["num_cpus"], verbose=self.ds.xrds.attrs["verbose"])
        ae_models.tf_init.init_float_type(float_type=self.ds.xrds.attrs["float_type"])




    @property
    def ds(self):
        return self.__ds

    @ds.setter
    def ds(self, ds):
        self.__ds = ds


    @property
    def loss_list(self):
        return self.__loss_list

    @loss_list.setter
    def loss_list(self, loss_list):
        self.__loss_list = loss_list



    @abstractmethod
    def run_fit(self):
         pass


    def run_autoencoder(self, **kwargs):
        self.run_fit(**kwargs)
        self.calc_X_pred()
        self.ds.init_pvalue_fc_z()
        self.xrds = self.ds.get_xrds()
        return self.xrds



    ##### prediction calculation steps

    ### hidden space
    def _pred_H(self, ae_input, E, cov_sample):
        H = np.matmul(ae_input, E)
        return H


    def _pred_X_trans(self, ae_input, E, D, b, cov_sample):
        y = np.matmul(self._pred_H(ae_input,E, cov_sample), D) # y: sample x gene
        y = y[:,0:len(b)]  # avoid cov_sample inclusion
        y_b = y + b
        y_b = utilis.float_limits.min_value_exp(y_b)
        return y_b

    def _pred_X(self, profile, ae_input, E, D, b, par_sample, cov_sample):
        y = self._pred_X_trans(ae_input, E, D, b, cov_sample)
        return rev_transform_ae_input(y, profile.ae_input_trans, par_sample=par_sample)


    def calc_X_pred(self):
        self.ds.X_pred = self._pred_X(self.ds.profile, self.ds.ae_input, self.ds.E, self.ds.D, self.ds.b, self.ds.par_sample, self.ds.cov_sample)
        self.ds.X_trans_pred = self._pred_X_trans(self.ds.ae_input, self.ds.E, self.ds.D, self.ds.b, self.ds.cov_sample)
        self._calc_X_pred()


    ### X value for pvalue calculation - raw or keep normalised
    def _calc_X_pred(self):
        if self.ds.profile.distribution.dis_name == "Dis_gaussian":
            self.ds.X_pred = self.ds.X_trans_pred
        elif self.ds.profile.distribution.dis_name == "Dis_neg_bin":
            self.ds.X_pred = self.ds.X_pred
        else:
            raise ValueError("distribution not found")


    def get_loss(self):
        self.calc_X_pred()
        ds_dis = self.ds.profile.distribution(X=self.ds.X, X_pred=self.ds.X_pred,
                                           par=self.ds.par_meas, parallel_iterations=self.ds.parallel_iterations)
        loss = ds_dis.get_loss()
        return loss






    ##### UPDATE PAR_MEAS STEPS #####
    # for OUTRIDER this is the update theta step


    def get_updated_par_meas(self, profile, X, X_pred, par_list, parallel_iterations=1):

        if profile.distribution.dis_name == "Dis_neg_bin":
            if par_list['init_step'] is True:
                par_meas  = robust_mom_theta(X, par_list['theta_range'][0], par_list['theta_range'][1])

            elif par_list['init_step'] is False:
                par_meas = self.update_par_meas_fmin(tf_neg_bin_loss, X=X, X_pred=X_pred,
                                                par_list = par_list['theta_range'], parallel_iterations=parallel_iterations)
        else:
            par_meas = np.zeros(shape=(X.shape[1], ))

        ### return np or tf
        return tf.convert_to_tensor(par_meas, dtype=X_pred.dtype)
        # if tf.is_tensor(X_pred):
        # else:
        #     return par_meas





    def update_par_meas_fmin(self, loss_func, X, X_pred, par_list, parallel_iterations):
            @tf.function
            def my_map(*args, **kwargs):
                return tf.map_fn(*args, **kwargs)

            X_pred_stacked = tf.transpose(tf.stack([X, X_pred], axis=1))
            par_meas = my_map(
                lambda row: tf_fminbound(
                    lambda x: loss_func(row[0, :], row[1, :], x),
                        x1=tf.constant(par_list[0]), x2=tf.constant(par_list[1])),
                        X_pred_stacked, parallel_iterations=parallel_iterations)

            return par_meas

