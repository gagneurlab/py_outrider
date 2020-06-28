import numpy as np
from distributions.transform_func import rev_trans_ae_input




def get_injected_outlier_gaussian(x,x_norm, norm_name, inj_freq, inj_mean, inj_sd, noise_factor, log, **kwargs):
    outlier_mask = np.random.choice([0,-1, 1], size=x_norm.shape, p=[1-inj_freq, inj_freq/2, inj_freq/2])

    if log:
        log_mean = np.log(inj_mean) if inj_mean != 0 else 0
        z_score = np.random.lognormal(mean=log_mean, sigma=np.log(inj_sd), size=x_norm.shape)
    else:
        z_score = np.random.normal(loc=inj_mean, scale=inj_sd, size=x_norm.shape)
    inj_values = np.abs(z_score) * noise_factor * np.nanstd(x_norm, ddof=1, axis=0)
    x_norm_outlier = inj_values * outlier_mask + x_norm

    ### avoid inj outlier to be too strong
    max_outlier_value = np.nanmin([10*np.nanmax(x), np.iinfo("int64").max])
    cond_value_too_big = rev_trans_ae_input(x_norm_outlier, norm_name, **kwargs) > max_outlier_value
    x_norm_outlier[cond_value_too_big] = x_norm[cond_value_too_big]
    outlier_mask[cond_value_too_big] = 0

    x_outlier = rev_trans_ae_input(x_norm_outlier, norm_name, **kwargs)
    return { "X_norm_outlier":x_norm_outlier, "X_outlier":x_outlier, "X_is_outlier":outlier_mask}













