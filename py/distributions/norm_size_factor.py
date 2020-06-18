import numpy as np





def calc_size_factor(counts):
    count_matrix = counts.values
    loggeomeans = np.mean(np.log(count_matrix), axis=0)
    sf = [_calc_size_factor_per_sample(x, loggeomeans, count_matrix) for x in count_matrix]
    return sf


def _calc_size_factor_per_sample(gene_list, loggeomeans, counts):
    sf_sample = np.exp( np.median((np.log(gene_list) - loggeomeans)[np.logical_and(np.isfinite(loggeomeans), counts[0, :] > 0)]))
    return sf_sample



def xrds_normalize_sf(xrds):
    sf = calc_size_factor(xrds["X"])
    xrds["size_factors"] = (("sample"), sf)

    def normalize_sf(counts, sf):
        x0 = np.log((counts + 1) / sf )
        x0_bias = np.mean(x0, axis=0)
        x1 = x0 - x0_bias
        return x1, x0_bias

    counts, bias = normalize_sf(xrds["X"], xrds["size_factors"])
    xrds["X_norm"] = ( ('sample','meas'), counts)
    xrds["X_center_bias"] = ( ('meas'), bias)




