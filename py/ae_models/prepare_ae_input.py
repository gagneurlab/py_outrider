
from distributions.norm_log2 import xrds_normalize_log2
from distributions.norm_size_factor import xrds_normalize_sf
from distributions.norm_none import xrds_normalize_none




def prepare_ae_input(xrds):
    ### inject outlier


    ### normalize data for ae model training
    if xrds.attrs["profile"].ae_input_norm == "sf":
        xrds_normalize_sf(xrds)
    elif xrds.attrs["profile"].ae_input_norm == "log2":
        xrds_normalize_log2(xrds)
    elif xrds.attrs["profile"].ae_input_norm == "none":
        xrds_normalize_none(xrds)


    ### inject noise






















