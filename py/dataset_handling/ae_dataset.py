import numpy as np
import tensorflow as tf    # 2.0.0
from tensorflow import math as tfm


import utilis.float_limits
from distributions.dis_gaussian import Dis_gaussian
from distributions.dis_neg_bin import Dis_neg_bin
from distributions.transform_func import transform_log2, transform_sf, transform_none
import utilis.stats_func as st
from distributions.transform_func import xrds_transform
from distributions.noise_func import get_injected_outlier_gaussian
from ae_models.loss_list import Loss_list

### accessing xarray matrices is pretty slow -> new class
### data container for all autoencoder data

class Ae_dataset():

    def __init__(self, xrds, inj_outlier=False):
        self.profile = xrds.attrs["profile"]

        self.xrds = xrds
        xrds_transform(self.xrds)

        if inj_outlier:
            self.inject_outlier(self.xrds, inj_freq=1e-3, inj_mean=3, inj_sd=1.6)
        self.inject_noise(self.xrds, inj_freq=1, inj_mean=0, inj_sd=1)

        ### change xrds
        # normalize_ae_input(self.xrds, self.xrds.attrs["profile"].ae_input_norm)

        self.initialize_ds(self.xrds)



    def initialize_ds(self, xrds):
        self.find_stat_used_X(xrds)

        self.X = xrds["X"].values
        self.X_norm = xrds["X_norm"].values
        self.X_norm_noise = xrds["X_norm_noise"].values
        self.X_center_bias = xrds["X_center_bias"].values
        self.cov_sample = xrds["cov_sample"].values if "cov_sample" in xrds else None
        self.par_sample = xrds["par_sample"].values if "par_sample" in xrds else None
        self.par_meas = xrds["par_meas"].values if "par_meas" in xrds else None
        # self.X = xrds["_X_stat_used"].values  # for pvalue and loss calculation

        self.X_pred = None  # for pvalue and loss calculation
        self.ae_input = None
        self.E = None
        self.D = None
        self.b = None
        self.H = None

        ### convert all to the same type
        self.X, self.X_norm, self.X_center_bias, self.cov_sample, self.par_sample, self.par_meas, self.X = \
            [ x.astype(xrds.attrs["float_type"], copy=False) if x is not None and x.dtype != xrds.attrs["float_type"] else x
              for x in [self.X, self.X_norm, self.X_center_bias, self.cov_sample, self.par_sample, self.par_meas, self.X] ]

        self.loss_list = Loss_list(conv_limit=0.0001, last_iter=3)

        self.parallel_iterations = xrds.attrs["num_cpus"]




    # ### values used for p-value calculation
    # def find_stat_used_X(self, xrds):
    #     if self.profile.distribution.dis_name == "Dis_neg_bin":
    #         xrds["_X_stat_used"] = xrds["X"]
    #     else:
    #         xrds["_X_stat_used"] = xrds["X_norm"] + xrds["X_center_bias"]



    def calc_pvalue(self):
        ds_dis = self.profile.distribution(X=self.X, X_pred=self.X_pred,
                                           par=self.par_meas, parallel_iterations=self.parallel_iterations)
        self.X_pvalue = ds_dis.get_pvalue()
        self.X_pvalue_adj = ds_dis.get_pvalue_adj()


    def init_pvalue_fc_z(self):
        self.calc_pvalue()
        self.X_log2fc = st.get_log2fc(self.X, self.X_pred)
        self.X_zscore = st.get_z_score(self.X_log2fc)



    def inject_noise(self, xrds, inj_freq, inj_mean, inj_sd, **kwargs):

        if "par_sample" in xrds:
            sf = xrds["par_sample"]

        inj_obj = get_injected_outlier_gaussian(x=xrds["X"].values, x_norm=xrds["X_norm"].values,
                                                norm_name=xrds.attrs["profile"].ae_input_trans,
                                                inj_freq=inj_freq, inj_mean=inj_mean, inj_sd=inj_sd,
                                                noise_factor=xrds.attrs["profile"].noise_factor, log=False, **kwargs)
        xrds["X_norm_noise"] = (('sample', 'meas'), inj_obj["X_norm_outlier"])
        xrds["X_noise"] = (('sample', 'meas'), inj_obj["X_outlier"])



    ### TODO avoid injection twice: if X_wo_outlier exists ..
    def inject_outlier(self, xrds,inj_freq, inj_mean, inj_sd, **kwargs):
        inj_obj = get_injected_outlier_gaussian(x=xrds["X"].values, x_norm=xrds["X_norm"].values,
                                                norm_name=xrds.attrs["profile"].ae_input_trans,
                                                inj_freq=inj_freq, inj_mean=inj_mean, inj_sd=inj_sd,
                                                noise_factor=1, log=True, **kwargs)
        xrds["X_wo_outlier"] = (('sample', 'meas'), xrds["X"])
        xrds["X"] = (('sample', 'meas'), inj_obj["X_outlier"])
        xrds["X_norm"] = (('sample', 'meas'), inj_obj["X_norm_outlier"])
        xrds["X_is_outlier"] = (('sample', 'meas'), inj_obj["X_is_outlier"])








    ## write everything into xrds
    def get_xrds(self):
        self.xrds["X_pred"] = (("sample", "meas"), self.X_pred)
        self.xrds["X_pvalue"] = (("sample", "meas"), self.X_pvalue)
        self.xrds["X_pvalue_adj"] = (("sample", "meas"), self.X_pvalue_adj)
        self.xrds["X_log2fc"] = (("sample", "meas"), self.X_log2fc)
        self.xrds["X_zscore"] = (("sample", "meas"), self.X_zscore)
        self.xrds["X_norm_pred"] = (("sample", "meas"), self.X_norm_pred)

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









