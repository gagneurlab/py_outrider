import warnings
from sklearn.decomposition import PCA
from nipals import nipals
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd

from .utils import tf_helper_func as tfh

warnings.simplefilter(action='ignore', category=FutureWarning)


class Encoder_AE():

    def __init__(self, encoding_dim, loss):
        self.E = None
        self.encoding_dim = encoding_dim
        self.loss = loss

    def get_encoder(self):
        if tf.is_tensor(self.E):
            return self.E.numpy()
        else:
            return self.E

    def init(self, x_in):
        e_pca = Encoder_PCA(self.encoding_dim, self.loss)
        e_pca.init(x_in)
        self.E = e_pca.get_encoder()

    @tf.function
    def encode(self, X):
        return Encoder_AE._encode(X, self.E)

    @staticmethod
    @tf.function
    def _encode(X, E):
        return tfh.tf_nan_matmul(X, E)

    def subset(self, features):
        self.E = self.E[features, :]
        return self

    @tf.function
    def loss_func_e(self, encoder_params, x_in, x_true, decoder, **kwargs):
        # print("Tracing loss_func_e with ... \nself = ", self,
        #       "\nencoder_params = ", encoder_params, "\nx_in = ", x_in,
        #       "\nx_true = ", x_true, "\ndecoder = ", decoder,
        #       "\ndecoder.x_na = ", decoder.x_na.shape,
        #        "\nkwargs = ", kwargs)
        E = tf.reshape(encoder_params, tf.shape(self.E))
        x_pred = decoder.decode(Encoder_AE._encode(x_in, E))[0]
        return self.loss(x_true=x_true, x_pred=x_pred, **kwargs)

    @tf.function
    def lbfgs_input(self, e, x_in, x_true, decoder, **kwargs):
        loss = self.loss_func_e(encoder_params=e, x_in=x_in, x_true=x_true,
                                decoder=decoder, **kwargs)
        gradients = tf.gradients(loss, e)[0]
        # e = tf.Variable(e)
        # with tf.GradientTape() as tape:
        #     loss = self.loss_func_e(encoder_params=e, x_in=x_in,
        #                             x_true=x_true, decoder=decoder, **kwargs)
        # gradients = tape.gradient(loss, e)
        # print(f"E Loss: {loss}")
        # print(f"E Gradients: {gradients}")
        return loss, tf.clip_by_value(gradients, -100., 100.)

    @staticmethod
    def _fit_lbfgs(e_init, loss, n_parallel):

        optim = tfp.optimizer.lbfgs_minimize(loss,
                                             initial_position=e_init,
                                             tolerance=1e-8,
                                             max_iterations=100,  # 500, # 150,
                                             num_correction_pairs=10,
                                             parallel_iterations=n_parallel)
        return optim.position

    def fit(self, x_in, x_true, decoder, optimizer, n_parallel, **kwargs):

        if optimizer == "lbfgs":

            e_init = tf.reshape(self.E, shape=[tf.size(self.E), ])
            loss = lambda x_e: self.lbfgs_input(e=x_e, x_in=x_in,
                                                x_true=x_true, decoder=decoder,
                                                **kwargs)

            new_E = Encoder_AE._fit_lbfgs(e_init, loss, n_parallel)
            self.E = tf.reshape(new_E, self.E.shape)

        return self.loss_func_e(tf.reshape(self.E, shape=[tf.size(self.E), ]),
                                x_in, x_true, decoder, **kwargs)


class Encoder_PCA():

    def __init__(self, encoding_dim, loss):
        self.E = None
        self.encoding_dim = encoding_dim
        self.loss = loss  # ignored in pca fit

    def get_encoder(self):
        if tf.is_tensor(self.E):
            return self.E.numpy()
        else:
            return self.E

    def init(self, x_in):
        # nipals if nan values are in matrix
        if np.isnan(x_in).any():
            try:
                # sometimes fails for big encoding dim for small samples
                nip = nipals.Nipals(x_in)
                nip.fit(ncomp=self.encoding_dim, maxiter=1500, tol=0.00001)
                self.E = np.transpose(nip.loadings.fillna(0).to_numpy())
                # for small encod_dim nan weights possible
            except:
                print(f"INFO: nipals failed for encod_dim {self.encoding_dim}"
                      ", using mean-imputed matrix and PCA for initialzation")
                # TODO emergency solution -> fix otherway (need starting point)
                fit_df = pd.DataFrame(x_in)
                x_in_imputed = fit_df.fillna(fit_df.mean()).to_numpy()
                self.init(x_in_imputed)
                return None
        else:
            pca = PCA(n_components=self.encoding_dim, svd_solver='full')
            pca.fit(x_in)  # X shape: n_samples x n_features
            self.E = pca.components_   # encod_dim x features
        self.E = tf.convert_to_tensor(np.transpose(self.E), dtype=x_in.dtype)

    @tf.function
    def encode(self, X):
        return Encoder_PCA._encode(X, self.E)

    @staticmethod
    @tf.function
    def _encode(X, E):
        return tfh.tf_nan_matmul(X, E)

    def subset(self, features):
        if tf.is_tensor(self.E):
            self.E = self.E.numpy()[features, :]
        else:
            self.E = self.E[features, :]
        return self

    @tf.function
    def loss_func_e(self, E, x_in, x_true, decoder, **kwargs):
        x_pred = decoder.decode(Encoder_PCA._encode(x_in, E))[0]
        return self.loss(x_true=x_true, x_pred=x_pred, **kwargs)

    def fit(self, x_in, x_true, decoder, optimizer, n_parallel, **kwargs):
        return self.loss_func_e(self.E, x_in, x_true, decoder, **kwargs)


LATENT_SPACE_MODELS = {'AE': Encoder_AE, 'PCA': Encoder_PCA}
