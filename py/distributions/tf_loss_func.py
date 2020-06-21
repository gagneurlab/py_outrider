import tensorflow as tf
from tensorflow import math as tfm
from utilis.float_limits import min_value_exp


@tf.function
def tf_neg_bin_loss(x_true, x_pred, theta):
    t1 = x_true * tfm.log(x_pred) + theta * tfm.log(theta)
    t2 = (x_true + theta) * tfm.log(x_pred + theta)
    t3 = tfm.lgamma(theta + x_true) - (tfm.lgamma(theta) + tfm.lgamma(x_true + 1))  # math: k! = exp(lgamma(k+1))

    ll = - tf.reduce_mean(t1 - t2 + t3)
    return ll


@tf.function
def tf_neg_bin_loss_E(e, D, b, x, x_norm, sf, theta, cov_sample):
    if x.shape == x_norm.shape:   # no covariates in encoding step
        if cov_sample is None:
            E_shape = tf.shape(tf.transpose(D))
        else:
            E_shape = (tf.shape(D)[1], (tf.shape(D)[0] - tf.shape(cov_sample)[1]))
        E = tf.reshape(e, E_shape)
        y = tf.matmul(tf.matmul(x, E), D) + b
    else:
        E_shape = (tf.shape(x_norm)[1], tf.shape(D)[0] - tf.shape(cov_sample)[1])
        E = tf.reshape(e, E_shape ) # sample+cov x encod_dim
        H = tf.concat([tf.matmul(x_norm, E), cov_sample], axis=1)
        y = tf.matmul(H, D) + b

    y = min_value_exp(y)

    t1 = x * (tf.expand_dims(tfm.log(sf), 1) + y)
    t2 = (x + theta) * (tf.expand_dims(tfm.log(sf), 1) + y
                        + tfm.log(1 + theta / (tf.expand_dims(sf, 1) * tfm.exp(y))))
    ll = tfm.reduce_mean(t1 - t2)
    return -ll


@tf.function
def tf_neg_bin_loss_D_single(H, c_i, b_and_D, sf, theta_i):
    b_i = b_and_D[0]
    D_i = b_and_D[1:]
    y = tf.squeeze( tf.matmul(H, tf.expand_dims(D_i,1)) + b_i )
    y = min_value_exp(y)

    t1 = c_i * (tfm.log(sf) + y)
    t2 = (c_i + theta_i) * ( tfm.log(sf) + y + tfm.log(1 + theta_i / (sf * tfm.exp(y))) )
    ll = tfm.reduce_mean(t1-t2)
    return -ll





