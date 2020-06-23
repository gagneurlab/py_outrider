import tensorflow as tf
from tensorflow import math as tfm
from utilis.float_limits import min_value_exp


@tf.function
def reshape_e_to_H(e, ae_input, X, D, cov_sample):
    if X.shape == ae_input.shape:   # no covariates in encoding step
        if cov_sample is None:
            E_shape = tf.shape(tf.transpose(D))
        else:
            E_shape = (tf.shape(D)[1], (tf.shape(D)[0] - tf.shape(cov_sample)[1]))
        E = tf.reshape(e, E_shape)
        H = tf.matmul(ae_input, E)  #
        # H = tf.concat([tf.matmul(ae_input, E), cov_sample], axis=1)  # uncomment if ae_bfgs_cov1
        return H
    else:
        E_shape = (tf.shape(ae_input)[1], tf.shape(D)[0] - tf.shape(cov_sample)[1])
        E = tf.reshape(e, E_shape ) # sample+cov x encod_dim
        H = tf.concat([tf.matmul(ae_input, E), cov_sample], axis=1)
        return H 




@tf.function
def tf_neg_bin_loss(x_true, x_pred, theta):
    t1 = x_true * tfm.log(x_pred) + theta * tfm.log(theta)
    t2 = (x_true + theta) * tfm.log(x_pred + theta)
    t3 = tfm.lgamma(theta + x_true) - (tfm.lgamma(theta) + tfm.lgamma(x_true + 1))  # math: k! = exp(lgamma(k+1))

    ll = - tf.reduce_mean(t1 - t2 + t3)
    return ll


@tf.function
def tf_neg_bin_loss_E(e, D, b, x, x_norm, sf, theta, cov_sample):
    H = reshape_e_to_H(e, x_norm, x, D, cov_sample)
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




@tf.function
def tf_gaus_loss_D_single(H, c_i, b_and_D, par_sample, par_meas_i):
    b_i = b_and_D[0]
    D_i = b_and_D[1:]
    y = tf.squeeze( tf.matmul(H, tf.expand_dims(D_i,1)) + b_i )
    return tf_gaus_loss(c_i, y)


@tf.function
def tf_gaus_loss_E(e, D, b, x, x_norm, par_sample, par_meas, cov_sample):
    H = reshape_e_to_H(e, x_norm, x, D, cov_sample)
    y = tf.matmul(H, D) + b
    return tf_gaus_loss(x, y)



@tf.function
def  tf_gaus_loss(x, x_pred):
    # PI = tf.constant(np.pi, dtype=x.dtype)
    #
    # mu = tf.reduce_mean(x_pred, axis=0)
    # sigma = tfm.reduce_std(x_pred, axis=0)
    #
    # ### short formula
    # var = sigma **2
    # ll = tfm.reduce_sum( -0.5 * tfm.log(2 * PI) - 0.5 * tfm.log(var) - 0.5 * (y - mu)**2 / var )
    # ll = ll / tf.size(y, out_type=y.dtype)
    # return -ll

    # ### ground truth tf implementation
    # # dist = tfp.distributions.Normal(loc=mu, scale=sigma)
    # # ll = tf.reduce_mean( dist.log_prob(c) )
    # # return tf.reduce_mean(tf.abs((y - x_pred)) )

    ### just taking equivalent RMSE
    return tf.keras.losses.MeanSquaredError()(x, x_pred)















