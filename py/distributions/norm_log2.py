import numpy as np




def xrds_normalize_log2(xrds):

    def normalize_log2(counts):
        x0 = np.log2(counts)
        x0_bias = np.mean(x0, axis=0)
        x1 = x0 - x0_bias
        return x1, x0_bias

    counts, bias = normalize_log2(xrds["X"])
    xrds["X_norm"] = ( ('sample','meas'), counts)
    xrds["X_center_bias"] = ( ('meas'), bias)


