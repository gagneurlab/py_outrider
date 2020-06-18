import tensorflow as tf
from tensorflow import math as tfm



@tf.function
def tf_neg_bin_loss(x_true, x_pred, theta):
    t1 = x_true * tfm.log(x_pred) + theta * tfm.log(theta)
    t2 = (x_true + theta) * tfm.log(x_pred + theta)
    t3 = tfm.lgamma(theta + x_true) - (tfm.lgamma(theta) + tfm.lgamma(x_true + 1))  # math: k! = exp(lgamma(k+1))

    ll = - tf.reduce_mean(t1 - t2 + t3)
    return ll