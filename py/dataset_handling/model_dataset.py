import numpy as np
import tensorflow as tf    # 2.0.0
import tensorflow_probability as tfp
from tensorflow import math as tfm

import utilis.stats_func as st
from ae_models.loss_list import Loss_list

import utilis

### accessing xarray matrices is pretty slow -> new class
### data container for all autoencoder data

class Model_dataset():

    def __init__(self, xrds):
        self.xrds = xrds
        self.profile = self.xrds.attrs["profile"]

        self.profile.data_trans.transform_xrds(self.xrds)

        # self.initialize_ds()



    ### must be called before model training
    def initialize_ds(self):
        # self.find_stat_used_X(xrds)

        self.X = self.xrds["X"].values
        self.X_pred = None  # for pvalue and loss calculation
        self.X_trans = self.xrds["X_trans"].values
        self.X_trans_noise = self.xrds["X_trans_noise"].values
        self.X_center_bias = self.xrds["X_center_bias"].values
        self.cov_sample = self.xrds["cov_sample"].values if "cov_sample" in self.xrds else None
        self.par_sample = self.xrds["par_sample"].values if "par_sample" in self.xrds else None
        self.par_meas = self.xrds["par_meas"].values if "par_meas" in self.xrds else None

        ### covariate consideration
        if self.cov_sample is not None:
            self.fit_input = np.concatenate([self.X_trans , self.cov_sample], axis=1)
            self.fit_input_noise = np.concatenate([self.X_trans_noise, self.cov_sample], axis=1)
        else:
            self.fit_input = self.X_trans
            self.fit_input_noise = self.X_trans_noise

        self.E = None
        self.D = None
        self.b = None
        self.H = None

        ### convert all to the same type
        self.X, self.X_trans, self.X_center_bias, self.cov_sample, self.par_sample, self.par_meas, self.X = \
            [ x.astype(self.xrds.attrs["float_type"], copy=False) if x is not None and x.dtype != self.xrds.attrs["float_type"] else x
              for x in [self.X, self.X_trans, self.X_center_bias, self.cov_sample, self.par_sample, self.par_meas, self.X] ]

        self.loss_list = Loss_list(conv_limit=0.00001, last_iter=3)
        self.parallel_iterations = self.xrds.attrs["num_cpus"]




    def calc_pvalue(self):
        ds_dis = self.profile.dis(X=self.X, X_pred=self.X_pred,
                                  par_meas=self.par_meas, parallel_iterations=self.parallel_iterations)
        self.X_pvalue = ds_dis.get_pvalue()
        self.X_pvalue_adj = ds_dis.get_pvalue_adj()


    def init_pvalue_fc_z(self):
        self.calc_pvalue()
        self.X_log2fc = st.get_log2fc(self.X, self.X_pred)
        self.X_zscore = st.get_z_score(self.X_log2fc)



    def inject_noise(self, inj_freq, inj_mean, inj_sd):
        inj_obj = self.profile.noise_dis.get_injected_outlier(X=self.xrds["X"].values, X_trans=self.xrds["X_trans"].values,
                                                        inj_freq=inj_freq, inj_mean=inj_mean, inj_sd=inj_sd, data_trans=self.profile.data_trans,
                                                        noise_factor=self.profile.noise_factor, par_sample=self.xrds["par_sample"])
        self.xrds["X_trans_noise"] = (('sample', 'meas'), inj_obj["X_trans_outlier"])
        self.xrds["X_noise"] = (('sample', 'meas'), inj_obj["X_outlier"])



    ### TODO avoid injection twice: if X_wo_outlier exists ..
    def inject_outlier(self, inj_freq, inj_mean, inj_sd):
        inj_obj = self.profile.outlier_dis.get_injected_outlier(X=self.xrds["X"].values, X_trans=self.xrds["X_trans"].values,
                                                        inj_freq=inj_freq, inj_mean=inj_mean, inj_sd=inj_sd, data_trans=self.profile.data_trans,
                                                        noise_factor=1, par_sample=self.xrds["par_sample"])
        self.xrds["X_wo_outlier"] = (('sample', 'meas'), self.xrds["X"])
        self.xrds["X"] = (('sample', 'meas'), inj_obj["X_outlier"])
        self.xrds["X_trans"] = (('sample', 'meas'), inj_obj["X_trans_outlier"])
        self.xrds["X_is_outlier"] = (('sample', 'meas'), inj_obj["X_is_outlier"])




    ##### prediction calculation steps
    @staticmethod   # need static otherwise self bug error
    def _pred_X_trans(H, D, b):
        # y = np.matmul(H, D)  # y: sample x gene
        # y = y[:, 0:len(b)]  # avoid cov_sample inclusion

        y = tf.matmul(H, D)  # y: sample x gene
        y = tf.gather(y, range(len(b)), axis=1)

        y_b = y + b
        y_b = utilis.float_limits.min_value_exp(y_b)
        return y_b


    def _pred_X(self, H, D, b, par_sample):
        y = Model_dataset._pred_X_trans(H, D, b)
        return self.profile.data_trans.rev_transform(y, par_sample=par_sample)


    def calc_X_pred(self):
        self.X_trans_pred = self._pred_X_trans(self.H, self.D, self.b)
        self.X_pred = self._pred_X(self.H, self.D, self.b, self.par_sample)


    def get_loss(self):
        ds_dis = self.profile.dis(X=self.X, X_pred=self.X_pred,
                                     par_meas=self.par_meas, parallel_iterations=self.parallel_iterations)
        loss = ds_dis.get_loss()
        return loss









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
        # self.xrds = self.xrds.drop_vars("_X_stat_used")

        return self.xrds









