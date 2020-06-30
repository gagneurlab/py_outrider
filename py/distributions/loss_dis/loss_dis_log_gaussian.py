import tensorflow as tf
from tensorflow import math as tfm

from fit_components.latent_space_fit.E_abstract import E_abstract

from distributions.loss_dis.loss_dis_abstract import Loss_dis_abstract


class Loss_dis_log_gaussian(Loss_dis_abstract):



    @staticmethod
    @tf.function
    def tf_loss(x, x_pred, **kwargs):
        # return tfm.log1p(tf.keras.losses.MeanSquaredError()(x, x_pred))

        x_log = tfm.log1p(x)
        x_pred_log = tfm.log1p(x_pred)
        gaus_loss = tf.keras.losses.MeanSquaredError()(x_log, x_pred_log)
        return gaus_loss























