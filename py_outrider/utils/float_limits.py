import tensorflow as tf
from tensorflow import math as tfm
import numpy as np


# small epsilon to avoid nan values
def eps():
    # return 1e-30
    return 1e-100  # TODO find other way to deal with this


def min_value_exp(t):
    if tf.is_tensor(t):
        if t.dtype is tf.float32:
            return tf.maximum(t, -100.)  # exp(x) for x<100 jumps to zero
        else:
            return tf.maximum(t, -700.)  # float64 used
    else:
        if t.dtype is np.float32:
            return np.maximum(t, -100.)  # exp(x) for x<100 jumps to zero
        else:
            return np.maximum(t, -700.)  # float64 used


def check_range_for_log(t):
    if tf.is_tensor(t):
        t = tf.where(tfm.is_inf(t), np.finfo(t.numpy().dtype).max / 1000, t)
        t = tf.where(t == 0., np.finfo(t.numpy().dtype).tiny * 1000, t)


def check_range_exp(t):
    if tf.is_tensor(t):
        t = tf.minimum(t, np.log(np.finfo(t.dtype.name).max / 1000))
        t = tf.maximum(t, np.log(np.finfo(t.dtype.name).tiny * 1000))
        return t
    else:
        t = np.minimum(t, np.log(np.finfo(t.dtype.name).max / 1000))
        t = np.maximum(t, np.log(np.finfo(t.dtype.name).tiny * 1000))
        return t


# avoid p-values == 0
def replace_zeroes_min(df):
    # min_nonzero = np.min(df[np.nonzero(df)])# trick to make it lowest p-value
    # df[df == 0] = min_nonzero
    df[df == 0] = 1e-100    # TODO not ideal
    return df
