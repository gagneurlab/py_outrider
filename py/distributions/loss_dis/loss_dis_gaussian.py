import tensorflow as tf
from tensorflow import math as tfm
from fit_components.latent_space_fit.E_abstract import E_abstract

from distributions.loss_dis.loss_dis_abstract import Loss_dis_abstract


class Loss_dis_gaussian(Loss_dis_abstract):



    @staticmethod
    @tf.function
    def tf_loss(x, x_pred, **kwargs):
        x_na = tfm.is_finite(x)
        return tf.keras.losses.MeanSquaredError()(tf.boolean_mask(x, x_na), tf.boolean_mask(x_pred, x_na))





















