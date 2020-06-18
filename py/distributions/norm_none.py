import numpy as np


### only center
def xrds_normalize_none(xrds):

    def normalize_none(counts):
        x0_bias = np.mean(counts, axis=0)
        x1 = counts - x0_bias
        return x1, x0_bias

    counts, bias = normalize_none(xrds["X"])
    xrds["X_norm"] = ( ('sample','meas'), counts)
    xrds["X_center_bias"] = ( ('meas'), bias)