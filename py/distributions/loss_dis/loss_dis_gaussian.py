import tensorflow as tf

from ae_models.encoder_fit.E_abstract import E_abstract

from distributions.loss_dis.loss_dis_abstract import Loss_dis_abstract


class Loss_dis_gaussian(Loss_dis_abstract):



    @staticmethod
    @tf.function
    def tf_loss(x, x_pred, **kwargs):
        return tf.keras.losses.MeanSquaredError()(x, x_pred)





















