from sklearn.decomposition import PCA
from nipals import nipals
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from .utils import tf_helper_func as tfh

class Encoder_AE():

    def __init__(self, encoding_dim, loss):
        self.E = None
        self.encoding_dim = encoding_dim
        self.loss = loss
        
    def get_encoder(self):
        return self.E.numpy()
    
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
        
    @tf.function
    def loss_func_e(self, encoder_params, x_in, x_true, decoder, **kwargs):
        E = tf.reshape(encoder_params, tf.shape(self.E))
        x_pred = decoder.decode(Encoder_AE._encode(x_in, E))[0]
        return self.loss(x_true=x_true, x_pred=x_pred, **kwargs)
        
    @tf.function
    def lbfgs_input(self, e, x_in, x_true, decoder, **kwargs):
        loss = self.loss_func_e(encoder_params=e, x_in=x_in, x_true=x_true, decoder=decoder, **kwargs)
        gradients = tf.gradients(loss, e)[0]
        # e = tf.Variable(e)
        # with tf.GradientTape() as tape:
        #     loss = self.loss_func_e(encoder_params=e, x_in=x_in, x_true=x_true, decoder=decoder, **kwargs)
        # gradients = tape.gradient(loss, e)
        # print(f"E Loss: {loss}")
        # print(f"E Gradients: {gradients}")
        return loss, tf.clip_by_value(gradients, -100., 100.)
        
    @staticmethod
    def _fit_lbfgs(e_init, loss, n_parallel):

        # def lbfgs_input(e):
        #     return tfp.math.value_and_gradient(
        #             lambda e: self.loss_func_e(e=e, adata=adata, loss_func=loss_func),
        #             e)
    
        # optim = tfp.optimizer.lbfgs_minimize(lambda e: self.lbfgs_input(e, x_in, x_true, decoder, kwargs),
        optim = tfp.optimizer.lbfgs_minimize(loss,
                                             initial_position=e_init, 
                                             tolerance=1e-8, 
                                             max_iterations=100, #500, #150,
                                             num_correction_pairs=10, 
                                             parallel_iterations=n_parallel)
        # print(f"E optim converged: {optim.converged}")
        return optim.position
    
    def fit(self, x_in, x_true, decoder, optimizer, n_parallel, **kwargs):
        
        if optimizer == "lbfgs":
            
            e_init = tf.reshape(self.E, shape=[tf.size(self.E), ])
            loss = lambda x_e: self.lbfgs_input(e=x_e, x_in=x_in, x_true=x_true, decoder=decoder, **kwargs)
            
            # print(f"E loss init: {self.loss_func_e(e_init, x_in, x_true, decoder, **kwargs)}")
            
            new_E = Encoder_AE._fit_lbfgs(e_init, loss, n_parallel)
            self.E = tf.reshape(new_E, self.E.shape)
            
            # print(f"E loss final: {self.loss_func_e(tf.reshape(self.E, shape=[tf.size(self.E), ]), x_in, x_true, decoder, **kwargs)}")
            
        return self.loss_func_e(tf.reshape(self.E, shape=[tf.size(self.E), ]), x_in, x_true, decoder, **kwargs)
            
        

class Encoder_PCA():

    def __init__(self, encoding_dim, loss):
        self.E = None
        self.encoding_dim = encoding_dim
        self.loss = loss # ignored in pca fit
        
    def get_encoder(self):
        return self.E
        
    def init(self, x_in):
        ### nipals if nan values are in matrix
        if np.isnan(x_in).any():
            nip = nipals.Nipals(x_in)
            nip.fit(ncomp=self.encoding_dim, maxiter=1500,tol=0.00001 )
            self.E = np.transpose(nip.loadings.fillna(0).to_numpy())  # for small encod_dim nan weights possible
        else:
            pca = PCA(n_components=self.encoding_dim, svd_solver='full')
            pca.fit(x_in) # X shape: n_samples x n_features
            self.E = pca.components_  # encod_dim x features
        self.E = tf.convert_to_tensor(np.transpose(self.E), dtype=x_in.dtype)
        
    @tf.function
    def encode(self, X):
        return tfh.tf_nan_matmul(X, self.E)
     
    # def encode(self, adata):
    #     latent = tfh.tf_nan_matmul(adata.X, self.E)
    #     adata.obsm['X_latent'] = latent
    #     adata.varm['E'] = self.E
    #     return adata
        
    def fit(self, **kwargs):
        pass
        
LATENT_SPACE_MODELS = {'AE': Encoder_AE, 'PCA': Encoder_PCA}
