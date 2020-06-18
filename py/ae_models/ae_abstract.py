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

class Ae_abstract(ABC):

    def __init__(self, xrds):
        self.xrds = xrds
        self.ae_ds = Ae_dataset(xrds)
        self.profile = xrds.attrs["profile"]

        self.loss_list = None

        ae_models.prepare_ae.init_tf_config(num_cpus=xrds.attrs["num_cpus"], verbose=xrds.attrs["verbose"])
        ae_models.prepare_ae.init_float_type(float_type=xrds.attrs["float_type"])

        self.parallel_iterations = xrds.attrs["num_cpus"]





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
            return tfm.exp(y) #* par_sample
        elif profile.ae_input_norm == "log2":
            y = self._pred_X_norm(ae_input, E, D, b)
            return tfm.pow(y,2)
        elif profile.ae_input_norm == "none":
            return self._pred_X_norm(ae_input, E, D, b)


    def get_X_norm_pred(self, ae_input, E, D, b):
        pred = self._pred_X_norm(ae_input, E, D, b)
        return pred

    def get_X_pred(self, ae_input, E, D, b, par_sample):
        pred = self._pred_X(self.profile, ae_input, E, D, b, par_sample)
        return pred

    ### X value for pvalue calculation - raw or keep normalised
    def get_X_true_pred(self, ae_input, E, D, b, par_sample):
        if self.profile.distribution == 'gaus':
            return self.get_X_norm_pred(ae_input, E,D,b)
        elif self.profile.distribution == 'neg_bin':
            return self.get_X_pred(ae_input, E, D, b, par_sample)




    ### activate after protein implementation again
    #
    # @abstractmethod
    # def get_encoder(self):
    #      pass
    #
    # @abstractmethod
    # def get_decoder(self):
    #      pass
    #
    # @abstractmethod
    # def get_hidden_space(self):
    #     pass


    #
    # def init_all(self):
    #     ae_input = self.ds_obj.get_ae_input()
    #
    #     if self.ds_obj.get_dataset_type() == 'gene':
    #         self.x = tf.convert_to_tensor(ae_input['counts'], dtype=self.ds_obj.float_type)
    #         self.par_samples = tf.convert_to_tensor(ae_input['sf'], dtype=self.ds_obj.float_type)
    #         self.x_norm, self.ae_model_bias = self.ds_obj.normalize_ae_input(self.x, self.par_samples)
    #         self.loss_func = neg_bin_loss_adam  ## only used by adam
    #         self.loss_func_E = neg_bin_loss_E
    #         self.loss_func_D = neg_bin_loss_D_single  # TODO change if use update_D with single vector loss function
    #         # self.loss_func_D = neg_bin_loss_D
    #
    #         # self.ae_model_bias = tf.reduce_mean(tfm.log(self.x+1), axis=0)  # OUTRIDER version  ### EDITED
    #
    #
    #
    #     ### TODO NOT IMPLEMENTED !!!!! - model changed
    #     elif self.ds_obj.get_dataset_type() == 'protein':
    #         self.x = tf.convert_to_tensor(ae_input['intensities'], dtype=self.ds_obj.float_type)
    #         self.par_samples = tf.convert_to_tensor(ae_input['batch_effects'], dtype=self.ds_obj.float_type)
    #         self.par_meas = None
    #         self.loss_func = tf.keras.losses.MSE
    #
    #     elif self.ds_obj.get_dataset_type() == 'gene_gaus':
    #         self.x = tf.convert_to_tensor(ae_input['counts'], dtype=self.ds_obj.float_type)
    #         self.x_norm, self.ae_model_bias = self.ds_obj.normalize_ae_input(self.x)
    #         self.par_samples = tf.zeros(shape=(self.x.shape[0], ) , dtype=self.ds_obj.float_type)
    #         self.par_meas = None
    #         self.loss_func = gaus_loss_adam ## only used by adam
    #         self.loss_func_E = gaus_loss_E
    #         # self.loss_func_D = gaus_loss_D
    #         self.loss_func_D = gaus_loss_D_single
    #         # self.ae_model_bias = tf.reduce_mean(self.x, axis=0) # centered substracted mean
    #
    #
    #     tf.keras.backend.set_floatx(self.x_norm.dtype.name)







    # def calc_pvalues(self):
    #     x_pred = self.x_pred
    #     ds_dis = self.ds_obj.get_distribution()(x_true=self.x, x_pred=x_pred,
    #                                             par=self.par_meas, float_type=self.ds_obj.float_type)
    #     pval = ds_dis.get_pval()
    #     pval_adj = ds_dis.get_pval_adj()
    #     return {'pval':pval, 'pval_adj':pval_adj}
    #
    #
    # def get_loss(self):
    #     x_pred = self.get_pred_y()
    #     ds_dis = self.ds_obj.get_distribution()(x_true=self.x, x_pred=x_pred,
    #                                           par=self.par_meas, float_type=self.ds_obj.float_type)
    #     loss = ds_dis.get_loss()
    #     return loss
    #
    #
    # def init_pval_fc_z(self):
    #     self.x_pred = self.get_pred_y()
    #     pval_dict = self.calc_pvalues()
    #     self.pval = pval_dict['pval']
    #     self.pval_adj = pval_dict['pval_adj']
    #     self.log2fc = get_log2fc(self.x, self.x_pred)
    #     self.z_score = get_z_score(self.log2fc)









    ##### UPDATE PAR_MEAS STEPS #####
    # for OUTRIDER this is the update theta step


    def get_updated_par_meas(self, X_true, X_true_pred, par_list, parallel_iterations=1):

        # if xrds.attrs == 'gene':  if distribution class neg_bin
        if True:
            if par_list['init_step'] is True:
                par_meas  = robust_mom_theta(X_true, par_list['theta_range'][0], par_list['theta_range'][1])

            elif par_list['init_step'] is False:
                par_meas = self.update_par_meas_fmin(tf_neg_bin_loss, x_true=X_true, x_pred=X_true_pred,
                                                par_list = par_list['theta_range'], parallel_iterations=parallel_iterations)
        else:
            par_meas = np.zeros(shape=(self.x.shape[1], ))

        ### return np or tf
        if tf.is_tensor(X_true_pred):
            return tf.convert_to_tensor(par_meas, dtype=X_true_pred.dtype)
        else:
            return par_meas





    def update_par_meas_fmin(self, loss_func, x_true, x_pred, par_list, parallel_iterations):
            @tf.function
            def my_map(*args, **kwargs):
                return tf.map_fn(*args, **kwargs)

            x_true_pred_stacked = tf.transpose(tf.stack([x_true, x_pred], axis=1))
            par_meas = my_map(
                lambda row: tf_fminbound(
                    lambda x: loss_func(row[0, :], row[1, :], x),
                    x1=tf.constant(par_list[0]), x2=tf.constant(par_list[1])),
                x_true_pred_stacked, parallel_iterations=parallel_iterations)

            return par_meas

