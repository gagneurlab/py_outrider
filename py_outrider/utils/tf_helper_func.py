import tensorflow as tf
from tensorflow import math as tfm
import numpy as np


@tf.function
def tf_nan_to_zero(t):
    return tf.where(tfm.is_nan(t), tf.zeros_like(t), t)


@tf.function
def tf_nan_matmul(a, b):
    # print("Tracing tf_nan_matmul with a = ", a, ", b = ", b)
    a_0 = tf_nan_to_zero(a)
    b_0 = tf_nan_to_zero(b)
    return tf.matmul(a_0, b_0)


@tf.function
def tf_set_nan(t, x_na):
    x_is_nan = tfm.logical_not(x_na)
    nan_t = tf.where(x_is_nan, tf.cast(np.nan, dtype=t.dtype), t)
    return nan_t


@tf.function
def tf_nan_func(func, **kwargs):
    """
    takes function with X as input parameter and applies function only on
    finite values,
    helpful for tf value calculation which can not deal with nan values
    :param func: function call with argument X
    :param kwargs: other arguments for func
    :return: executed func output with nan values
    """
    mask = tfm.is_finite(kwargs["X"])
    empty_t = tf.cast(tf.fill(mask.shape, np.nan), dtype=kwargs["X"].dtype)

    for i in kwargs:
        # workaround of tf.rank(kwargs[i]) > 0, avoid scalar value in mask
        if kwargs[i].shape != ():
            # keep only finite
            kwargs[i] = tf.boolean_mask(kwargs[i], tfm.is_finite(kwargs[i]))
    res_func = func(**kwargs)

    full_t = tf.tensor_scatter_nd_update(empty_t, tf.where(mask), res_func)
    return full_t
