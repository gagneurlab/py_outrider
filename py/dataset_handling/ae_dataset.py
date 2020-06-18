import numpy as np
import tensorflow as tf    # 2.0.0
from tensorflow import math as tfm


import utilis.float_limits
from distributions.dis_gaussian import Dis_gaussian
from distributions.dis_neg_bin import Dis_neg_bin
from distributions.norm_log2 import xrds_normalize_log2
from distributions.norm_none import xrds_normalize_none
from distributions.norm_size_factor import xrds_normalize_sf


### accessing xarray matrices is pretty slow -> new class
### data container for all autoencoder data

class Ae_dataset():

    def __init__(self, xrds):
        self.profile = xrds.attrs["profile"]
        self.xrds = xrds

        ### TODO
        ### inject outlier
        ### inject noise

        ### change xrds
        self.normalize_ae_input(xrds)
        self.find_stat_used_X(xrds)

        self.initialize_ds(xrds)



    def initialize_ds(self, xrds):
        self.X_norm = xrds["X_norm"].values
        self.X_center_bias = xrds["X_center_bias"].values
        self.cov_sample = xrds["cov_sample"].values if "cov_sample" in xrds else None
        self.par_sample = xrds["par_sample"].values if "par_sample" in xrds else None
        self.par_meas = xrds["par_meas"].values if "par_meas" in xrds else None
        self.X_true = xrds["_X_stat_used"].values  # for pvalue and loss calculation

        self.X_true_pred = None  # for pvalue and loss calculation
        self.ae_input = None
        self.E = None
        self.D = None
        self.b = None

        self.parallel_iterations = xrds.attrs["num_cpus"]





    ### normalize data for ae model training
    def normalize_ae_input(self, xrds):
        if xrds.attrs["profile"].ae_input_norm == "sf":
            xrds_normalize_sf(xrds)
        elif xrds.attrs["profile"].ae_input_norm == "log2":
            xrds_normalize_log2(xrds)
        elif xrds.attrs["profile"].ae_input_norm == "none":
            xrds_normalize_none(xrds)


    def find_stat_used_X(self, xrds):
        if self.profile.distribution.dis_name == "Dis_neg_bin":
            xrds["_X_stat_used"] = xrds["X"]
        else:
            xrds["_X_stat_used"] = xrds["X_norm"]






  ##### prediction calculation steps
    def _pred_X_norm(self, ae_input, E, D, b):
        y = np.matmul(np.matmul(ae_input, E), D)
        y_b = y + b
        y_b = utilis.float_limits.min_value_exp(y_b)
        return y_b

    def _pred_X(self, profile, ae_input, E, D, b, par_sample):
        if profile.ae_input_norm == "sf":
            y = self._pred_X_norm(ae_input, E, D, b)
            return tfm.exp(y) * tf.expand_dims(par_sample,1)
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
        if self.profile.distribution.dis_name == "Dis_gaussian":
            return self.get_X_norm_pred()
        elif self.profile.distribution.dis_name == "Dis_neg_bin":
            return self.get_X_pred()
        else:
            raise ValueError("distribution not found")


    def get_loss(self):
        self.X_true_pred = self.get_X_true_pred()
        ds_dis = self.profile.distribution(X_true=self.X_true, X_pred=self.X_true_pred,
                                           par=self.par_meas, parallel_iterations=self.parallel_iterations)
        loss = ds_dis.get_loss()
        return loss














    def return_xrds(self):
        # self.xrds["pred"] = 5
        None
        ## write everything into xrds

