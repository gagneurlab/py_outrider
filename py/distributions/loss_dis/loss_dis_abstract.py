from abc import ABC, abstractmethod
import tensorflow as tf
from ae_models.encoder_fit.E_abstract import E_abstract


class Loss_dis_abstract(ABC):



    @staticmethod
    @abstractmethod
    def tf_loss(self):
         pass

    # @staticmethod
    # @abstractmethod
    # def tf_loss_D(self):
    #      pass
    #
    # @staticmethod
    # @abstractmethod
    # def tf_loss_D_single(self):
    #      pass
    #
    # @staticmethod
    # @abstractmethod
    # def tf_loss_E(self):
    #     pass


    @classmethod
    @tf.function
    def tf_loss_E(cls, e, D, b, x, x_trans, cov_sample,data_trans, **kwargs):
        H = E_abstract.reshape_e_to_H(e=e, ae_input=x_trans, X=x, D=D, cov_sample=cov_sample)
        y = tf.matmul(H, D) + b
        x_pred = data_trans.rev_transform(y=y, **kwargs)
        return cls.tf_loss(x, x_pred, **kwargs)


    @classmethod
    @tf.function
    def tf_loss_D_single(cls, H, x_i, b_and_D,data_trans, **kwargs):
        b_i = b_and_D[0]
        D_i = b_and_D[1:]
        y = tf.squeeze( tf.matmul(H, tf.expand_dims(D_i,1)) + b_i )
        x_pred = data_trans.rev_transform(y=y, **kwargs)
        return cls.tf_loss(x_i, x_pred)


    @classmethod
    @tf.function
    def tf_loss_D(cls, H, x , b_and_D,data_trans, **kwargs):
        b_and_D = tf.reshape(b_and_D, [H.shape[1] + 1, x.shape[1]])
        b = b_and_D[0, :]
        D = b_and_D[1:, :]

        y = tf.transpose(tf.matmul(H, D) + b)
        x_pred = data_trans.rev_transform(y=y, **kwargs)
        return cls.tf_loss(x, x_pred, **kwargs)








