import numpy as np
from distributions.transform_func import rev_transform_ae_input




def get_injected_outlier_gaussian(X,X_trans, norm_name, inj_freq, inj_mean, inj_sd, noise_factor, log, par_sample):
    outlier_mask = np.random.choice([0,-1, 1], size=X_trans.shape, p=[1-inj_freq, inj_freq/2, inj_freq/2])

    if log:
        log_mean = np.log(inj_mean) if inj_mean != 0 else 0
        z_score = np.random.lognormal(mean=log_mean, sigma=np.log(inj_sd), size=X_trans.shape)
    else:
        z_score = np.random.normal(loc=inj_mean, scale=inj_sd, size=X_trans.shape)
    inj_values = np.abs(z_score) * noise_factor * np.nanstd(X_trans, ddof=1, axis=0)
    X_trans_outlier = inj_values * outlier_mask + X_trans

    ### avoid inj outlier to be too strong
    max_outlier_value = np.nanmin([10*np.nanmax(X), np.iinfo("int64").max])
    cond_value_too_big = rev_transform_ae_input(X_trans_outlier, norm_name, par_sample) > max_outlier_value
    X_trans_outlier[cond_value_too_big] = X_trans[cond_value_too_big]
    outlier_mask[cond_value_too_big] = 0

    X_outlier = rev_transform_ae_input(X_trans_outlier, norm_name, par_sample)
    return { "X_trans_outlier":X_trans_outlier, "X_outlier":X_outlier, "X_is_outlier":outlier_mask}













