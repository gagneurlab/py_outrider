import numpy as np
import tensorflow as tf    # 2.0.0
from tensorflow import math as tfm


import utilis.float_limits
from distributions.dis_gaussian import Dis_gaussian
from distributions.dis_neg_bin import Dis_neg_bin
from distributions.norm_func import xrds_normalize_log2, xrds_normalize_sf, xrds_normalize_none
import utilis.stats_func as st


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

        self.initialize_ds(xrds)



    def initialize_ds(self, xrds):
        self.find_stat_used_X(xrds)

        self.X = xrds["X"].values
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
        self.H = None

        ### convert all to the same type
        self.X, self.X_norm, self.X_center_bias, self.cov_sample, self.par_sample, self.par_meas, self.X_true = \
            [ x.astype(xrds.attrs["float_type"], copy=False) if x is not None and x.dtype != xrds.attrs["float_type"] else x
              for x in [self.X, self.X_norm, self.X_center_bias, self.cov_sample, self.par_sample, self.par_meas, self.X_true] ]


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








    def calc_pvalue(self):
        ds_dis = self.profile.distribution(X_true=self.X_true, X_pred=self.X_true_pred,
                                           par=self.par_meas, parallel_iterations=self.parallel_iterations)
        self.X_pvalue = ds_dis.get_pvalue()
        self.X_pvalue_adj = ds_dis.get_pvalue_adj()


    def init_pvalue_fc_z(self):
        self.calc_pvalue()
        self.X_log2fc = st.get_log2fc(self.X, self.X_pred)
        self.X_zscore = st.get_z_score(self.X_log2fc)






    ## write everything into xrds
    def get_xrds(self):
        self.xrds["X_pred"] = (("sample", "meas"), self.X_pred)
        self.xrds["X_pvalue"] = (("sample", "meas"), self.X_pvalue)
        self.xrds["X_pvalue_adj"] = (("sample", "meas"), self.X_pvalue_adj)
        self.xrds["X_log2fc"] = (("sample", "meas"), self.X_log2fc)

        if self.par_sample is not None:
            self.xrds["par_sample"] = (("sample"), self.par_sample)
        if self.par_meas is not None:
            self.xrds["par_meas"] = (("meas"), self.par_meas)

        ### ae model parameter
        self.xrds["encoder_weights"] = (("meas_cov","encod_dim"), self.E)
        self.xrds["decoder_weights"] = (("encod_dim_cov", "meas"), self.D)
        self.xrds["decoder_bias"] = (("meas"), self.b)

        ### init new coordinates
        if self.E.shape[0] != len(self.xrds.coords["meas"]):
            self.xrds.coords["meas_cov"] = np.concatenate( [self.xrds.coords["meas"], self.xrds.coords["cov_sample_col"]])
        else:
            self.xrds.coords["meas_cov"] = self.xrds.coords["meas"]

        if self.D.shape[0] != len(self.xrds.coords["encod_dim"]):
            self.xrds.coords["encod_dim_cov"] = np.concatenate( [self.xrds.coords["encod_dim"], self.xrds.coords["cov_sample_col"]])
        else:
            self.xrds.coords["encod_dim_cov"] = self.xrds.coords["encod_dim"]


        ### remove unncessary
        self.xrds = self.xrds.drop_vars("_X_stat_used")

        return self.xrds









