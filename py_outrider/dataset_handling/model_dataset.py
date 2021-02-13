import numpy as np
import tensorflow as tf    # 2.0.0
import tensorflow_probability as tfp
from tensorflow import math as tfm
import copy

from ..fit_components.loss_list import Loss_list
from .. import utils
from ..utils import stats_func as st
from ..utils import tf_helper_func as tfh

### accessing xarray matrices is pretty slow -> new class
### data container for all autoencoder data

class Model_dataset():

    def __init__(self, xrds):
        self.xrds = xrds
        self.profile = self.xrds.attrs["profile"]
        self.profile.data_trans.transform_xrds(self.xrds)
        self.encod_dim = self.xrds.attrs["encod_dim"]

        # self.initialize_ds()

    @property
    def encod_dim(self):
        return self.__encod_dim

    @encod_dim.setter
    def encod_dim(self, encod_dim):
        if encod_dim is not None:
            encod_dim = int(min(self.xrds["X"].shape[1], encod_dim))
        self.xrds.attrs["encod_dim"] = encod_dim
        self.__encod_dim = encod_dim



    ### must be called before model training - gets called by model training
    def initialize_ds(self):

        self.X = self.xrds["X"].values
        self.X_na = self.xrds["X_na"].values
        self.X_pred = None  # for pvalue and loss calculation
        self.X_trans = self.xrds["X_trans"].values
        self.X_trans_noise = self.xrds["X_trans_noise"].values
        self.X_center_bias = self.xrds["X_center_bias"].values
        self.cov_sample = self.xrds["cov_sample"].values if "cov_sample" in self.xrds else None
        self.sizefactors = self.xrds["sizefactors"].values if "sizefactors" in self.xrds else None
        self.dispersions = self.xrds["dispersions"].values if "dispersions" in self.xrds else None

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

        ### convert all parameters to the same float type, cannot convert to function without copying
        self.X, self.X_trans, self.X_trans_noise, self.X_center_bias, self.cov_sample,\
        self.sizefactors, self.dispersions, self.X, self.fit_input, self.fit_input_noise = \
            [ x.astype(self.xrds.attrs["float_type"], copy=False) if x is not None and x.dtype != self.xrds.attrs["float_type"] else x
              for x in [self.X, self.X_trans, self.X_trans_noise, self.X_center_bias, self.cov_sample,
                        self.sizefactors, self.dispersions, self.X, self.fit_input, self.fit_input_noise] ]

        self.loss_list = Loss_list()
        self.parallel_iterations = self.xrds.attrs["num_cpus"]







    def calc_pvalue(self):
        ds_dis = self.profile.dis(X=self.X, X_pred=self.X_pred,
                                  dispersions=self.dispersions, parallel_iterations=self.parallel_iterations)
        self.X_pvalue = ds_dis.get_pvalue()
        self.X_pvalue_adj = ds_dis.get_pvalue_adj()




    def init_pvalue_fc_z(self):
        self.calc_pvalue()
        self.X_logfc = self.profile.data_trans.get_logfc(X_trans=self.X_trans, X_trans_pred=self.X_trans_pred, sizefactors=self.sizefactors )
        self.X_zscore = st.get_z_score(self.X_logfc)



    def inject_noise(self, inj_freq, inj_mean, inj_sd):
        inj_obj = self.profile.noise_dis.get_injected_outlier(X=self.xrds["X"].values, X_trans=self.xrds["X_trans"].values,
                                                              X_center_bias=self.xrds["X_center_bias"].values,
                                                        inj_freq=inj_freq, inj_mean=inj_mean, inj_sd=inj_sd, data_trans=self.profile.data_trans,
                                                        noise_factor=self.profile.noise_factor, seed=self.xrds.attrs["seed"], sizefactors=self.xrds["sizefactors"])
        self.xrds["X_trans_noise"] = (('sample', 'feature'), inj_obj["X_trans_outlier"])
        self.xrds["X_noise"] = (('sample', 'feature'), inj_obj["X_outlier"])



    def inject_outlier(self, inj_freq, inj_mean, inj_sd):
        self._remove_inj_outlier()
        inj_obj = self.profile.outlier_dis.get_injected_outlier(X=self.xrds["X"].values, X_trans=self.xrds["X_trans"].values,
                                                                X_center_bias=self.xrds["X_center_bias"].values,
                                                        inj_freq=inj_freq, inj_mean=inj_mean, inj_sd=inj_sd, data_trans=self.profile.data_trans,
                                                        noise_factor=1, sizefactors=self.xrds["sizefactors"], seed=self.xrds.attrs["seed"])


        ##TODO force at least one injection otherwise nan prec-rec value - bad practice -> add cancel
        ### makes sure there are injected outliers in measurement table
        tmp_seed = self.xrds.attrs["seed"]
        while np.nansum(inj_obj["X_is_outlier"]) == 0:
            if tmp_seed is not None:
                tmp_seed+=1
            print("repeat outlier injection")
            inj_obj = self.profile.outlier_dis.get_injected_outlier( X=self.xrds["X"].values, X_trans=self.xrds["X_trans"].values,
                                                                    X_center_bias=self.xrds["X_center_bias"].values,
                                                                    inj_freq=inj_freq, inj_mean=inj_mean, inj_sd=inj_sd,
                                                                    data_trans=self.profile.data_trans,
                                                                    noise_factor=1, sizefactors=self.xrds["sizefactors"], seed=tmp_seed)

        self.xrds["X_wo_outlier"] = (('sample', 'feature'), self.xrds["X"])
        self.xrds["X"] = (('sample', 'feature'), inj_obj["X_outlier"])
        self.xrds["X_trans_wo_outlier"] = (('sample', 'feature'), self.xrds["X_trans"])
        self.xrds["X_trans"] = (('sample', 'feature'), inj_obj["X_trans_outlier"])
        self.xrds["X_is_outlier"] = (('sample', 'feature'), inj_obj["X_is_outlier"])

        # TODO X_center_bias again as it changes slightly with outlier injection


    def _remove_inj_outlier(self):
        self.xrds["X"] = self.xrds["X_wo_outlier"] if "X_wo_outlier" in self.xrds else self.xrds["X"]
        self.xrds["X_trans"] = self.xrds["X_trans_wo_outlier"] if "X_trans_wo_outlier" in self.xrds else self.xrds["X_trans"]




    ##### prediction calculation steps
    #@tf.function
    @staticmethod   # need static otherwise self bug error
    def _pred_X_trans(X_na, H, D, b, data_trans):
        # y = np.matmul(H, D)  # y: sample x gene
        # y = y[:, 0:len(b)]  # avoid cov_sample inclusion
        y = tf.matmul(H, D)  # y: sample x gene
        y = tf.gather(y, range(len(b)), axis=1)

        y_b = y + b
        #y_b = py_outrider.utils.float_limits.min_value_exp(y_b) # only meaningful for log transformation
        y_b = data_trans.check_range_trans(y_b)
        y_b = tfh.tf_set_nan(y_b, X_na)
        return y_b


    def _pred_X(self,X_na, H, D, b, sizefactors):
        y = Model_dataset._pred_X_trans(X_na=X_na, H=H, D=D, b=b, data_trans=self.profile.data_trans)
        return self.profile.data_trans.rev_transform(y, sizefactors=sizefactors)


    def calc_X_pred(self):
        self.X_trans_pred = self._pred_X_trans(X_na=self.X_na, H=self.H, D=self.D, b=self.b, data_trans=self.profile.data_trans)
        self.X_pred = self._pred_X(X_na=self.X_na, H=self.H, D=self.D, b=self.b, sizefactors=self.sizefactors)


    def get_loss(self):
        ds_dis = self.profile.dis(X=self.X, X_pred=self.X_pred,
                                     dispersions=self.dispersions, parallel_iterations=self.parallel_iterations)
        loss = ds_dis.get_loss()
        return loss





    def get_xrds(self):
        """
        writes whole model_dataset object back into a xarray dataset, ready to output
        :return: xarray dataset
        """
        self.xrds.coords["encod_dim"] =  ["q_" + str(d) for d in range(self.encod_dim)]

        self.xrds["X_pred"] = (("sample", "feature"), self.X_pred)
        self.xrds["X_pvalue"] = (("sample", "feature"), self.X_pvalue)
        self.xrds["X_pvalue_adj"] = (("sample", "feature"), self.X_pvalue_adj)
        self.xrds["X_logfc"] = (("sample", "feature"), self.X_logfc)
        self.xrds["X_zscore"] = (("sample", "feature"), self.X_zscore)
        self.xrds["X_trans_pred"] = (("sample", "feature"), self.X_trans_pred)

        # ae_norm_values = self.X / self.X_pred * tfm.reduce_mean(self.X, axis=0)  # TODO DOES NOT WORK WITH NAN !!!
        ae_norm_values = self.X / self.X_pred * np.nanmean(self.X, axis=0)

        self.xrds["X_norm"] = (("sample", "feature"), ae_norm_values)

        if self.sizefactors is not None:
            self.xrds["sizefactors"] = (("sample"), self.sizefactors)
        if self.dispersions is not None:
            self.xrds["dispersions"] = (("feature"), self.dispersions)

        self.xrds["encoder_weights"] = (("feature_cov","encod_dim"), self.E)
        self.xrds["decoder_weights"] = (("encod_dim_cov", "feature"), self.D)
        self.xrds["decoder_bias"] = (("feature"), self.b)

        ### init new coordinates
        if self.E.shape[0] != len(self.xrds.coords["feature"]):
            self.xrds.coords["feature_cov"] = np.concatenate( [self.xrds.coords["feature"], self.xrds.coords["cov_sample_col"]])
        else:
            self.xrds.coords["feature_cov"] = self.xrds.coords["feature"]

        if self.D.shape[0] != len(self.xrds.coords["encod_dim"]):
            self.xrds.coords["encod_dim_cov"] = np.concatenate( [self.xrds.coords["encod_dim"], self.xrds.coords["cov_sample_col"]])
        else:
            self.xrds.coords["encod_dim_cov"] = self.xrds.coords["encod_dim"]

        if self.xrds.verbose:
            print('loss_summary')
            print(self.loss_list.loss_summary)

        return self.xrds



    def print_dataset_shapes(self):

        def get_shape(tensor):
            if tensor is None:
                return None
            else:
                return tensor.shape

        print("### model_dataset shapes ###")
        print(f"  fit_input: {get_shape(self.fit_input)}")
        print(f"  E: {get_shape(self.E)}")
        print(f"  D: {get_shape(self.D)}")
        print(f"  b: {get_shape(self.b)}")
        print(f"  H: {get_shape(self.H)}")
        print(f"  X_trans: {get_shape(self.X_trans)}")
        print(f"  X: {get_shape(self.X)}")
        print(f"  X_pred: {get_shape(self.X_pred)}")
        print(f"  cov_sample: {get_shape(self.cov_sample)}")
        print(f"  dispersions: {get_shape(self.dispersions)}")

    def subset_samples(self, sample_idx):
        # create subsetted object
        sub = copy.copy(self)
        sub.xrds = sub.xrds.isel(sample=sample_idx)
        sub.X = sub.X[sample_idx,:]
        sub.X_na = sub.X_na[sample_idx,:]
        sub.X_pred = tf.gather(sub.X_pred, indices=sample_idx, axis=0)
        sub.X_trans = sub.X_trans[sample_idx,:]
        sub.X_trans_noise = sub.X_trans_noise[sample_idx,:]
        sub.fit_input = sub.fit_input[sample_idx,:]
        sub.fit_input_noise = sub.fit_input_noise[sample_idx,:]
        sub.sizefactors = sub.sizefactors[sample_idx]
        sub.H = tf.gather(sub.H, indices=sample_idx, axis=0)
        
        return sub

    
        


