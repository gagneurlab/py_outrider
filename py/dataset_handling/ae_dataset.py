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

    def __init__(self, xrds):
        self.xrds = xrds
        self.profile = self.xrds.attrs["profile"]
        xrds_transform(self.xrds)

        # self.initialize_ds()



    ### must be called before model training
    def initialize_ds(self):
        # self.find_stat_used_X(xrds)

        self.X = self.xrds["X"].values
        self.X_trans = self.xrds["X_trans"].values
        self.X_trans_noise = self.xrds["X_trans_noise"].values
        self.X_center_bias = self.xrds["X_center_bias"].values
        self.cov_sample = self.xrds["cov_sample"].values if "cov_sample" in self.xrds else None
        self.par_sample = self.xrds["par_sample"].values if "par_sample" in self.xrds else None
        self.par_meas = self.xrds["par_meas"].values if "par_meas" in self.xrds else None
        # self.X = self.xrds["_X_stat_used"].values  # for pvalue and loss calculation

        self.X_pred = None  # for pvalue and loss calculation
        self.ae_input = None
        self.E = None
        self.D = None
        self.b = None
        self.H = None

        ### convert all to the same type
        self.X, self.X_trans, self.X_center_bias, self.cov_sample, self.par_sample, self.par_meas, self.X = \
            [ x.astype(self.xrds.attrs["float_type"], copy=False) if x is not None and x.dtype != self.xrds.attrs["float_type"] else x
              for x in [self.X, self.X_trans, self.X_center_bias, self.cov_sample, self.par_sample, self.par_meas, self.X] ]

        self.loss_list = Loss_list(conv_limit=0.0001, last_iter=3)


        self.parallel_iterations = self.xrds.attrs["num_cpus"]




    # ### values used for p-value calculation
    # def find_stat_used_X(self, xrds):
    #     if self.profile.distribution.dis_name == "Dis_neg_bin":
    #         xrds["_X_stat_used"] = xrds["X"]
    #     else:
    #         xrds["_X_stat_used"] = xrds["X_trans"] + xrds["X_center_bias"]



    def calc_pvalue(self):
        ds_dis = self.profile.distribution(X=self.X, X_pred=self.X_pred,
                                           par=self.par_meas, parallel_iterations=self.parallel_iterations)
        self.X_pvalue = ds_dis.get_pvalue()
        self.X_pvalue_adj = ds_dis.get_pvalue_adj()


    def init_pvalue_fc_z(self):
        self.calc_pvalue()
        self.X_log2fc = st.get_log2fc(self.X, self.X_pred)
        self.X_zscore = st.get_z_score(self.X_log2fc)



    def inject_noise(self, inj_freq, inj_mean, inj_sd):
        inj_obj = get_injected_outlier_gaussian(X=self.xrds["X"].values, X_trans=self.xrds["X_trans"].values,
                                                norm_name=self.profile.ae_input_trans,
                                                inj_freq=inj_freq, inj_mean=inj_mean, inj_sd=inj_sd,
                                                noise_factor=self.profile.noise_factor, log=False, par_sample=self.xrds["par_sample"])
        self.xrds["X_trans_noise"] = (('sample', 'meas'), inj_obj["X_trans_outlier"])
        self.xrds["X_noise"] = (('sample', 'meas'), inj_obj["X_outlier"])



    ### TODO avoid injection twice: if X_wo_outlier exists ..
    def inject_outlier(self, inj_freq, inj_mean, inj_sd):
        inj_obj = get_injected_outlier_gaussian(X=self.xrds["X"].values, X_trans=self.xrds["X_trans"].values,
                                                norm_name=self.profile.ae_input_trans,
                                                inj_freq=inj_freq, inj_mean=inj_mean, inj_sd=inj_sd,
                                                noise_factor=1, log=True, par_sample=self.xrds["par_sample"])
        self.xrds["X_wo_outlier"] = (('sample', 'meas'), self.xrds["X"])
        self.xrds["X"] = (('sample', 'meas'), inj_obj["X_outlier"])
        self.xrds["X_trans"] = (('sample', 'meas'), inj_obj["X_trans_outlier"])
        self.xrds["X_is_outlier"] = (('sample', 'meas'), inj_obj["X_is_outlier"])








    ## write everything into xrds
    def get_xrds(self):
        self.xrds["X_pred"] = (("sample", "meas"), self.X_pred)
        self.xrds["X_pvalue"] = (("sample", "meas"), self.X_pvalue)
        self.xrds["X_pvalue_adj"] = (("sample", "meas"), self.X_pvalue_adj)
        self.xrds["X_log2fc"] = (("sample", "meas"), self.X_log2fc)
        self.xrds["X_zscore"] = (("sample", "meas"), self.X_zscore)
        self.xrds["X_trans_pred"] = (("sample", "meas"), self.X_trans_pred)

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









