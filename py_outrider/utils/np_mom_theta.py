import numpy as np


# first estimation of theta step
def robust_mom_theta(x, theta_min, theta_max, mu_min=0.1):
    mue = trim_mean(x, proportiontocut=0.125, axis=0)
    mue = np.maximum(mue, mu_min)

    se = (x - np.array([mue] * x.shape[0])) ** 2
    see = trim_mean(se, proportiontocut=0.125, axis=0)
    ve = 1.51 * see

    th = mue ** 2 / (ve - mue)
    th[th < 0] = theta_max

    re_th = np.maximum(theta_min, np.minimum(theta_max, th))
    return re_th


# from https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/stats/stats.py#L3180-L3247 # noqa
def trim_mean(a, proportiontocut, axis=0):
    a = np.asarray(a)

    if a.size == 0:
        return np.nan

    if axis is None:
        a = a.ravel()
        axis = 0

    nobs = a.shape[axis]
    lowercut = int(proportiontocut * nobs)
    uppercut = nobs - lowercut
    if (lowercut > uppercut):
        raise ValueError("Proportion too big.")

    atmp = np.partition(a, (lowercut, uppercut - 1), axis)

    sl = [slice(None)] * atmp.ndim
    sl[axis] = slice(lowercut, uppercut)
    return np.mean(atmp[tuple(sl)], axis=axis)
