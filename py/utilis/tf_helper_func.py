import tensorflow as tf
from tensorflow import math as tfm
import numpy as np


def tf_nan_to_zero(t):
    return tf.where(tfm.is_nan(t), tf.zeros_like(t), t)



def tf_nan_matmul(a, b):
    a_0 = tf_nan_to_zero(a)
    b_0 = tf_nan_to_zero(b)
    return tf.matmul(a_0, b_0)


def tf_set_nan(t, x_na):
    x_is_nan = tfm.logical_not(x_na)
    return tf.where(x_is_nan, np.nan, t)









































