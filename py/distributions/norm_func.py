import numpy as np


def xrds_normalize_none(xrds):

    def normalize_none(counts):
        x0_bias = np.mean(counts, axis=0)
        x1 = counts - x0_bias
        return x1, x0_bias

    counts, bias = normalize_none(xrds["X"])
    xrds["X_norm"] = ( ('sample','meas'), counts)
    xrds["X_center_bias"] = ( ('meas'), bias)



def xrds_normalize_log2(xrds):

    def normalize_log2(counts):
        x0 = np.log2(counts+1)
        x0_bias = np.mean(x0, axis=0)
        x1 = x0 - x0_bias
        return x1, x0_bias

    counts, bias = normalize_log2(xrds["X"])
    xrds["X_norm"] = ( ('sample','meas'), counts)
    xrds["X_center_bias"] = ( ('meas'), bias)




def xrds_normalize_sf(xrds):
    sf = calc_size_factor(xrds["X"])
    xrds["par_sample"] = (("sample"), sf)

    def normalize_sf(counts, sf):
        x0 = np.log((counts + 1) / sf )
        x0_bias = np.mean(x0, axis=0)
        x1 = x0 - x0_bias
        return x1, x0_bias

    counts, bias = normalize_sf(xrds["X"], xrds["par_sample"])
    xrds["X_norm"] = ( ('sample','meas'), counts)
    xrds["X_center_bias"] = ( ('meas'), bias)


def _calc_size_factor_per_sample(gene_list, loggeomeans, counts):
    sf_sample = np.exp( np.median((np.log(gene_list) - loggeomeans)[np.logical_and(np.isfinite(loggeomeans), counts[0, :] > 0)]))
    return sf_sample


def calc_size_factor(counts):
    count_matrix = counts.values
    loggeomeans = np.mean(np.log(count_matrix), axis=0)
    sf = [_calc_size_factor_per_sample(x, loggeomeans, count_matrix) for x in count_matrix]
    return sf


