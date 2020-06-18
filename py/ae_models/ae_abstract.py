from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf    # 2.0.0
from tensorflow import math as tfm
import scipy



# from ae_tf_functions.neg_bin_update_theta import tf_fminbound_minimize_theta, fminbound_minimize_theta, tf_theta_loss_per_gene
# from ae_tf_functions.neg_bin_loss import neg_bin_loss_adam
# from ae_tf_functions.neg_bin_update_D import neg_bin_loss_D_single, neg_bin_loss_D
# from ae_tf_functions.neg_bin_update_E import neg_bin_loss_E
# from ae_tf_functions.gaus_update import gaus_loss_D, gaus_loss_E, gaus_loss_D_single
# from ae_tf_functions.gaus_loss import gaus_loss_adam
# from statistic.fc_z_score import get_log2fc, get_z_score
# from utilis_methods.tf_fminbound import tf_fminbound
#
# from ae_tf_functions import vst_transform

import ae_models.prepare_ae
import utilis.float_limits
from utilis.tf_fminbound import tf_fminbound
from distributions.tf_loss_func import tf_neg_bin_loss
from utilis.tf_mom_theta import robust_mom_theta
from dataset_handling.ae_dataset import Ae_dataset
from distributions.dis_neg_bin import Dis_neg_bin
from distributions.dis_gaussian import Dis_gaussian


class Ae_abstract(ABC):

    def __init__(self, xrds):
        self.xrds = xrds
        self.ae_ds = Ae_dataset(xrds)
        self.profile = xrds.attrs["profile"]

        self.loss_list = None

        ae_models.prepare_ae.init_tf_config(num_cpus=xrds.attrs["num_cpus"], verbose=xrds.attrs["verbose"])
        ae_models.prepare_ae.init_float_type(float_type=xrds.attrs["float_type"])

        self.parallel_iterations = xrds.attrs["num_cpus"]

        self.initialize_new(xrds)


    def initialize_new(self, xrds):
        self.par_sample = xrds["par_sample"].values if "par_sample" in xrds else None
        self.par_meas = xrds["par_meas"].values if "par_meas" in xrds else None


    @property
    def ae_ds(self):
        return self.__ae_ds

    @ae_ds.setter
    def ae_ds(self, ae_ds):
        self.__ae_ds = ae_ds

    @property
    def profile(self):
        return self.__profile

    @profile.setter
    def profile(self, profile):
        self.__profile = profile

    @property
    def loss_list(self):
        return self.__loss_list

    @loss_list.setter
    def loss_list(self, loss_list):
        self.__loss_list = loss_list

    @property
    def parallel_iterations(self):
        return self.__parallel_iterations

    @parallel_iterations.setter
    def parallel_iterations(self, parallel_iterations):
        self.__parallel_iterations = parallel_iterations


    @property
    def E(self):
        return self.__E

    @E.setter
    def E(self, E):
        self.__E = E

    @property
    def D(self):
        return self.__D

    @D.setter
    def D(self, D):
        self.__D = D

    @property
    def b(self):
        return self.__b

    @b.setter
    def b(self, b):
        self.__b = b


    @property
    def par_sample(self):
        return self.__par_sample

    @par_sample.setter
    def par_sample(self, par_sample):
        self.__par_sample = par_sample

    @property
    def par_meas(self):
        return self.__par_meas

    @par_meas.setter
    def par_meas(self, par_meas):
        self.__par_meas = par_meas





    @abstractmethod
    def run_autoencoder(self):
         pass



    ##### prediction calculation steps
    def _pred_X_norm(self, ae_input, E, D, b):
        y = np.matmul(np.matmul(ae_input, E), D)
        y_b = y + b
        y_b = utilis.float_limits.min_value_exp(y_b)
        return y_b

    def _pred_X(self, profile, ae_input, E, D, b, par_sample):
        if profile.ae_input_norm == "sf":
            y = self._pred_X_norm(ae_input, E, D, b)
            return tfm.exp(y) * par_sample
        elif profile.ae_input_norm == "log2":
            y = self._pred_X_norm(ae_input, E, D, b)
            return tfm.pow(y,2)
        elif profile.ae_input_norm == "none":
            return self._pred_X_norm(ae_input, E, D, b)


    def get_X_norm_pred(self):
        pred = self._pred_X_norm(self.ae_input, self.E, self.D, self.b)
        return pred

    def get_X_pred(self):
        pred = self._pred_X(self.profile, self.ae_input, self.E, self.D, self.b, self.par_sample)
        return pred


    ### X value for pvalue calculation - raw or keep normalised
    def get_X_true_pred(self):
        print('5')
        if isinstance(self.profile.distribution, Dis_gaussian):
            return self.get_X_norm_pred()
        elif isinstance(self.profile.distribution, Dis_neg_bin):
            return self.get_X_pred()




    def get_loss(self):
        X_true_pred = self.get_X_true_pred()
        print(X_true_pred.shape)
        print(self.ae_ds.X_true.shape)
        ds_dis = self.profile.distribution(X_true=self.ae_ds.X_true, X_pred=X_true_pred,
                                           par=self.par_meas, parallel_iterations=self.parallel_iterations)
        loss = ds_dis.get_loss()
        return loss



    # def calc_pvalues(self):
    #     X_pred = self.X_pred
    #     ds_dis = self.ds_obj.get_distribution()(X_true=self.x, X_pred=X_pred,
    #                                             par=self.par_meas, float_type=self.ds_obj.float_type)
    #     pval = ds_dis.get_pval()
    #     pval_adj = ds_dis.get_pval_adj()
    #     return {'pval':pval, 'pval_adj':pval_adj}
    #
    #
    #
    #
    # def init_pval_fc_z(self):
    #     self.X_pred = self.get_pred_y()
    #     pval_dict = self.calc_pvalues()
    #     self.pval = pval_dict['pval']
    #     self.pval_adj = pval_dict['pval_adj']
    #     self.log2fc = get_log2fc(self.x, self.X_pred)
    #     self.z_score = get_z_score(self.log2fc)









    ##### UPDATE PAR_MEAS STEPS #####
    # for OUTRIDER this is the update theta step


    def get_updated_par_meas(self, X_true, X_true_pred, par_list, parallel_iterations=1):

        if isinstance(self.profile.distribution, Dis_neg_bin):
            if par_list['init_step'] is True:
                par_meas  = robust_mom_theta(X_true, par_list['theta_range'][0], par_list['theta_range'][1])

            elif par_list['init_step'] is False:
                par_meas = self.update_par_meas_fmin(tf_neg_bin_loss, X_true=X_true, X_pred=X_true_pred,
                                                par_list = par_list['theta_range'], parallel_iterations=parallel_iterations)
        else:
            par_meas = np.zeros(shape=(X_true.shape[1], ))

        ### return np or tf
        if tf.is_tensor(X_true_pred):
            return tf.convert_to_tensor(par_meas, dtype=X_true_pred.dtype)
        else:
            return par_meas





    def update_par_meas_fmin(self, loss_func, X_true, X_pred, par_list, parallel_iterations):
            @tf.function
            def my_map(*args, **kwargs):
                return tf.map_fn(*args, **kwargs)

            X_true_pred_stacked = tf.transpose(tf.stack([X_true, X_pred], axis=1))
            par_meas = my_map(
                lambda row: tf_fminbound(
                    lambda x: loss_func(row[0, :], row[1, :], x),
                        x1=tf.constant(par_list[0]), x2=tf.constant(par_list[1])),
                        X_true_pred_stacked, parallel_iterations=parallel_iterations)

            return par_meas

