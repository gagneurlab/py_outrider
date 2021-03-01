import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import copy

from .preprocess import rev_trans_tf
from .utils import tf_helper_func as tfh


class Decoder_AE():

    def __init__(self, loss):
        self.loss = loss
        self.D = None
        self.bias = None
        self.cov = None
        self.sf = None
        self.rev_trans = 'none'

    def get_decoder(self):
        D, bias = self.D, self.bias
        if tf.is_tensor(D):
            D = D.numpy()
        if tf.is_tensor(bias):
            bias = bias.numpy()
        return D, bias

    def copy(self):
        return copy.copy(self)

    def init(self, encoder, x_na, feature_means=None, sf=1, trans_func='none',
             cov=None):
        nr_cov_oneh_cols = cov.shape[1] if cov is not None else 0
        D = tf.transpose(encoder.E)
        if nr_cov_oneh_cols > 0:
            D = D[:, :-nr_cov_oneh_cols]
            cov_init_weights = np.zeros(shape=(nr_cov_oneh_cols, D.shape[1]))
            D = np.concatenate([D, cov_init_weights], axis=0)
            D = tf.convert_to_tensor(D)
        self.D = D
        if feature_means is not None:
            self.bias = feature_means
        else:
            self.bias = np.zeros(encoder.E.shape[0])
        self.bias = tf.convert_to_tensor(self.bias)

        self.sf = sf
        self.rev_trans = trans_func
        self.x_na = x_na
        self.cov = cov

    @tf.function
    def decode(self, X_latent):
        return Decoder_AE._decode(X_latent, self.D, self.bias, self.sf,
                                  self.rev_trans, self.x_na, self.cov)

    @staticmethod
    @tf.function
    def _decode(X_latent, D, bias, sf, trans_fun, x_na, cov):
        if cov is not None:
            X_latent = tf.concat([X_latent, cov], axis=1)
        prediction_no_trans = tf.matmul(X_latent, D)
        prediction_no_trans = tf.math.add(prediction_no_trans, bias)
        prediction = rev_trans_tf(prediction_no_trans, sf, trans_fun)
        prediction = tfh.tf_set_nan(prediction, x_na)
        return prediction, prediction_no_trans

    @tf.function
    def loss_func_d(self, decoder_params, x_latent, x_true, **kwargs):
        if "x_na" in kwargs:
            x_na = kwargs.pop("x_na")
        else:
            x_na = self.x_na
        x_pred = Decoder_AE._decode(x_latent, D=decoder_params[0],
                                    bias=decoder_params[1], sf=self.sf,
                                    trans_fun=self.rev_trans, x_na=x_na,
                                    cov=self.cov)[0]
        return self.loss(x_true, x_pred, **kwargs)

    @tf.function
    def lbfgs_loss_and_gradients(self, b_and_D, x_latent, x_true,
                                 single_feature=False, **kwargs):
        if single_feature is True:
            x_b = b_and_D[0]
            x_D = tf.expand_dims(b_and_D[1:], 1)
        else:
            b_and_D = tf.reshape(b_and_D, [self.D.shape[0] + 1,
                                           self.D.shape[1]])
            x_b = b_and_D[0, :]
            x_D = b_and_D[1:, :]
        loss = self.loss_func_d(decoder_params=[x_D, x_b], x_latent=x_latent,
                                x_true=x_true, **kwargs)
        gradients = tf.gradients(loss, b_and_D)[0]
        return loss, tf.clip_by_value(tf.reshape(gradients, [-1]), -100., 100.)
        # x_D = tf.Variable(x_d)
        # x_b = tf.Variable(x_b)
        # with tf.GradientTape() as tape:
        #    loss = self.loss_func_d(decoder_params=[x_D, x_b],
        #                            x_latent=x_latent, x_true=x_true,
        #                            **kwargs)

        # gradients = tape.gradient(loss, [x_b, x_D])  # works in eager mode
        # print(f"D Loss: {loss}")
        # print(f"D Gradients: {gradients}")
        # gradients = tf.concat([tf.expand_dims(gradients[0], 0),
        #                                       gradients[1]], axis=0)

    @staticmethod
    def _fit_lbfgs(b_and_D_init, loss_and_gradients, n_parallel):

        optim = tfp.optimizer.lbfgs_minimize(loss_and_gradients,
                                             initial_position=b_and_D_init,
                                             tolerance=1e-8,
                                             max_iterations=100,  # 300, # 100,
                                             num_correction_pairs=10,
                                             parallel_iterations=n_parallel)
        return optim.position

    @staticmethod
    @tf.function(experimental_relax_shapes=True)
    def _fit_lbfgs_by_feature(input_tensors, loss_gradients, x_latent):
        # input_tensors = (D, b, x_true, dispersions)

        # print("Tracing single D fit with tensors = ", input_tensors)
        D_i, b_i, x_i, x_na_i = input_tensors[0:4]
        dispersions_i = input_tensors[4] if len(input_tensors) == 5 else None

        b_and_D = tf.concat([tf.expand_dims(b_i, 0), D_i], axis=0)

        @tf.function
        def loss_gradients_feature(b_and_D):
            return loss_gradients(b_and_D,
                                  x_latent=x_latent,
                                  x_true=tf.expand_dims(x_i, 1),
                                  dispersions=dispersions_i,
                                  x_na=tf.expand_dims(x_na_i, 1),
                                  single_feature=True)

        optim = tfp.optimizer.lbfgs_minimize(loss_gradients_feature,
                                             initial_position=b_and_D,
                                             tolerance=1e-6,
                                             max_iterations=50,  # 100
                                             num_correction_pairs=10)
        return optim.position

    @staticmethod
    @tf.function(experimental_relax_shapes=True)
    def get_optim_results_feature(tensors_to_vectorize, fit_func_lbfgs_feature,
                                  n_parallel):
        # print("Tracing get_optim_results_feature with ... \ntensors = ",
        #       tensors_to_vectorize, "\nfit_func = ", fit_func_lbfgs_feature,
        #       "\nn_parallel = ", n_parallel)
        # return tf.vectorized_map(fit_func_lbfgs_feature,
        #                          tensors_to_vectorize)
        return tf.map_fn(fit_func_lbfgs_feature,
                         tensors_to_vectorize,
                         fn_output_signature=tensors_to_vectorize[2].dtype,
                         parallel_iterations=n_parallel)

    def fit(self, x_latent, x_true, optimizer, n_parallel,
            parallelize_by_feature, **kwargs):

        if optimizer == "lbfgs":
            if parallelize_by_feature is True:

                @tf.function
                def fit_func_lbfgs_feature(t):
                    return self._fit_lbfgs_by_feature(
                        input_tensors=t,
                        loss_gradients=self.lbfgs_loss_and_gradients,
                        x_latent=x_latent)

                if kwargs["dispersions"] is not None:
                    # tensors_to_vectorize = (D, b, x_true, x_na, dispersions)
                    tensors_to_vectorize = (tf.transpose(self.D), self.bias,
                                            tf.transpose(x_true),
                                            tf.transpose(self.x_na),
                                            kwargs["dispersions"])
                else:
                    # tensors_to_vectorize = (D, b, x_true, x_na)
                    tensors_to_vectorize = (tf.transpose(self.D), self.bias,
                                            tf.transpose(x_true),
                                            tf.transpose(self.x_na))
                optim_results = self.get_optim_results_feature(
                    tensors_to_vectorize,
                    fit_func_lbfgs_feature=fit_func_lbfgs_feature,
                    n_parallel=n_parallel)
                b_and_D_out = tf.transpose(optim_results)

            else:
                b_and_D_init = tf.concat([tf.expand_dims(self.bias, 0),
                                          self.D],
                                         axis=0)
                b_and_D_init = tf.reshape(b_and_D_init, [-1])  # flatten

                def loss_and_gradients(b_and_D):
                    return self.lbfgs_loss_and_gradients(b_and_D=b_and_D,
                                                         x_latent=x_latent,
                                                         x_true=x_true,
                                                         single_feature=False,
                                                         **kwargs)

                optim_results = Decoder_AE._fit_lbfgs(b_and_D_init,
                                                      loss_and_gradients,
                                                      n_parallel)
                b_and_D_out = tf.reshape(optim_results, [self.D.shape[0] + 1,
                                                         self.D.shape[1]])

            self.bias = tf.convert_to_tensor(b_and_D_out[0, :])
            self.D = tf.convert_to_tensor(b_and_D_out[1:, :])

        return self.loss_func_d([self.D, self.bias], x_latent, x_true,
                                **kwargs)


class Decoder_PCA():

    def __init__(self, loss):
        self.D = None
        self.bias = None
        self.loss = loss

    def get_decoder(self):
        D, bias = self.D, self.bias
        if tf.is_tensor(D):
            D = D.numpy()
        if tf.is_tensor(bias):
            bias = bias.numpy()
        return D, bias

    def init(self, encoder, x_na, feature_means=None, sf=1, trans_func='none',
             cov=None):
        nr_cov_oneh_cols = cov.shape[1] if cov is not None else 0
        D = tf.transpose(encoder.E)
        if nr_cov_oneh_cols > 0:
            D = D[:, :-nr_cov_oneh_cols]
            cov_init_weights = np.zeros(shape=(nr_cov_oneh_cols, D.shape[1]))
            D = np.concatenate([D, cov_init_weights], axis=0)
        self.D = D
        if feature_means is not None:
            self.bias = feature_means
        else:
            self.bias = np.zeros(encoder.E.shape[0])
        self.bias = tf.convert_to_tensor(self.bias)

        self.sf = sf
        self.rev_trans = trans_func
        self.x_na = x_na
        self.cov = cov

    @tf.function
    def decode(self, X_latent):
        return Decoder_PCA._decode(X_latent, self.D, self.bias, self.sf,
                                   self.rev_trans, self.x_na, self.cov)

    @staticmethod
    @tf.function
    def _decode(X_latent, D, bias, sf, trans_fun, x_na, cov):
        if cov is not None:
            X_latent = tf.concat([X_latent, cov], axis=1)
        prediction_no_trans = tf.matmul(X_latent, D)
        prediction_no_trans = tf.math.add(prediction_no_trans, bias)
        prediction = rev_trans_tf(prediction_no_trans, sf, trans_fun)
        prediction = tfh.tf_set_nan(prediction, x_na)
        return prediction, prediction_no_trans

    @tf.function
    def loss_func_d(self, decoder_params, x_latent, x_true, **kwargs):
        x_pred = Decoder_AE._decode(x_latent, D=decoder_params[0],
                                    bias=decoder_params[1], sf=self.sf,
                                    trans_fun=self.rev_trans, x_na=self.x_na,
                                    cov=self.cov)[0]
        return self.loss(x_true, x_pred, **kwargs)

    def fit(self, x_latent, x_true, optimizer, n_parallel, **kwargs):
        return self.loss_func_d([self.D, self.bias], x_latent, x_true,
                                **kwargs)

    def copy(self):
        return copy.copy(self)


DECODER_MODELS = {'AE': Decoder_AE, 'PCA': Decoder_PCA}
