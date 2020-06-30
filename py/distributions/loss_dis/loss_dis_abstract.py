from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow import math as tfm
from fit_components.latent_space_fit.E_abstract import E_abstract



class Loss_dis_abstract(ABC):



    # @tf.function
    @staticmethod
    @abstractmethod
    def tf_loss(x, x_pred, par_meas, par_sample):
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
    def tf_loss_E(cls, e, D, b, x, x_trans, cov_sample, data_trans, **kwargs):
        _, H = E_abstract.reshape_e_to_H(e=e, ae_input=x_trans, X=x, D=D, cov_sample=cov_sample)

        y = tf.matmul(H, D) + b
        x_pred = data_trans.rev_transform(y=y, **kwargs)
        return cls.tf_loss(x=x, x_pred=x_pred, **kwargs)


    # @tf.function
    @classmethod
    def tf_loss_D_single(cls, H, x_i, b_and_D, data_trans, **kwargs):
        b_i = b_and_D[0]
        D_i = b_and_D[1:]
        y = tf.squeeze( tf.matmul(H, tf.expand_dims(D_i,1)) + b_i )
        x_pred = data_trans.rev_transform(y=y, **kwargs)

        return cls.tf_loss(x=x_i, x_pred=x_pred, **kwargs)


    # @tf.function
    @classmethod
    def tf_loss_D(cls, H, x , b_and_D, data_trans, **kwargs):
        b_and_D = tf.reshape(b_and_D, [H.shape[1] + 1, x.shape[1]])
        b = b_and_D[0, :]
        D = b_and_D[1:, :]

        y = tf.matmul(H, D) + b
        x_pred = data_trans.rev_transform(y=y, **kwargs)
        # return Loss_dis_abstract.tf_loss(x=x, x_pred=x_pred, **kwargs)

        # print('D_Loss')
        # print(cls.tf_loss)
        l = cls.tf_loss(x=x, x_pred=x_pred, **kwargs)
        # print(l)

        return cls.tf_loss(x=x, x_pred=x_pred, **kwargs)


    #
    # @staticmethod
    # @tf.function
    # def tf_loss(x, x_pred, par_meas, **kwargs):
    #     theta = par_meas
    #
    #     t1 = x * tfm.log(x_pred) + theta * tfm.log(theta)
    #     t2 = (x + theta) * tfm.log(x_pred + theta)
    #     t3 = tfm.lgamma(theta + x) - (tfm.lgamma(theta) + tfm.lgamma(x + 1))  # math: k! = exp(lgamma(k+1))
    #
    #     ll = - tf.reduce_mean(t1 - t2 + t3)
    #     return ll
    #
    #
    #

