import tensorflow as tf    # 2.0.0
import numpy as np


### small epsilon to avoid nan values
def eps():
    #return 1e-30
    return 1e-100  ## TODO find other way to deal with this



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


### avoid p-values == 0
def replace_zeroes_min(df):
    #min_nonzero = np.min(df[np.nonzero(df)])  # trick to make it lowest p-value -> shitty approach
    # df[df == 0] = min_nonzero
    df[df == 0] = 1e-100    # TODO not ideal
    return df