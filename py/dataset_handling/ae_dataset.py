from distributions.norm_log2 import xrds_normalize_log2
from distributions.norm_size_factor import xrds_normalize_sf
from distributions.norm_none import xrds_normalize_none



### accessing xarray matrices is pretty slow -> new class

class Ae_dataset():

    def __init__(self, xrds):
        ### TODO
        ### inject outlier
        ### inject noise

        self.normalize_ae_input(xrds)
        self.find_stat_used_X(xrds)

        self.X_norm = xrds["X_norm"].values
        self.X_center_bias = xrds["X_center_bias"].values
        self.cov_sample = xrds["cov_sample"].values if "cov_sample" in xrds else None
        self.par_sample = xrds["par_sample"].values if "par_sample" in xrds else None
        self.par_meas = xrds["par_meas"].values if "par_meas" in xrds else None
        self.X_true = xrds["_X_stat_used"]  # for pvalue and loss calculation
        self.E = None
        self.D = None
        self.b = None




    ### normalize data for ae model training
    def normalize_ae_input(self, xrds):
        if xrds.attrs["profile"].ae_input_norm == "sf":
            xrds_normalize_sf(xrds)
        elif xrds.attrs["profile"].ae_input_norm == "log2":
            xrds_normalize_log2(xrds)
        elif xrds.attrs["profile"].ae_input_norm == "none":
            xrds_normalize_none(xrds)


    def find_stat_used_X(self,xrds):
        if xrds.attrs["profile"].distribution== "neg_bin":
            xrds["_X_stat_used"] = xrds["X"]
        else:
            xrds["_X_stat_used"] = xrds["X_norm"]






    def return_xrds(self):
        None
        ## write everything into xrds
















