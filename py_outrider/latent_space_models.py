from sklearn.decomposition import PCA
from nipals import nipals
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from .utils import tf_helper_func as tfh

class Encoder_AE():

    def __init__(self, encoding_dim):
        self.E = None
        self.encoding_dim = encoding_dim
        
    def get_encocer(self):
        return self.E
    
    def init(self,adata):
        e_pca = Encoder_PCA(self.encoding_dim)
        e_pca.init(adata)
        self.E = e_pca.get_encoder()

        
    def encode(self, adata):
        adata.obsm['X_latent'] = tfh.tf_nan_matmul(adata.X, self.E)
        adata.varm['E'] = self.E
        return adata
        
    @tf.function
    def loss_func_e(self, e, adata, loss_func):
        new_enc = Encoder_AE(self.encoding_dim)
        new_enc.E = tf.reshape(e, tf.shape(self.E))
        return loss_func(encoder=new_enc, adata=adata)
        
    @tf.function
    def fit(self, adata, loss_func, optimizer, n_parallel):
        if optimizer == "lbfgs":
            e = tf.reshape(self.E, shape=[tf.size(self.E), ])

            def lbfgs_input(e):
                loss = self.loss_func_e(e=e, adata=adata, loss_func=loss_func)
                gradients = tf.gradients(loss, e)[0]
                return loss, tf.clip_by_value(gradients, -100., 100.)
    
            optim = tfp.optimizer.lbfgs_minimize(lbfgs_input, 
                                                 initial_position=e, 
                                                 tolerance=1e-8, 
                                                 max_iterations=100, #500, #150,
                                                 num_correction_pairs=10, 
                                                 parallel_iterations=n_parallel)
    
            self.E = tf.reshape(optim.position, tf.shape(self.E))
            
        

class Encoder_PCA():

    def __init__(self, encoding_dim):
        self.E = None
        self.encoding_dim = encoding_dim
        
    def get_encoder(self):
        return self.E
        
    def init(self, adata):
        self.fit(adata, loss_func=None, optimizer=None, n_parallel=None)
        
    
    def encode(self, adata):
        latent = tfh.tf_nan_matmul(adata.X, self.E)
        print(f"Latent space: {latent}")
        adata.obsm['X_latent'] = latent
        adata.varm['E'] = self.E
        return adata
        
    def fit(self, adata, loss_func, optimizer, n_parallel):
        ### nipals if nan values are in matrix
        if np.isnan(adata.X).any():
            nip = nipals.Nipals(adata.X)
            nip.fit(ncomp=self.encoding_dim, maxiter=1500,tol=0.00001 )
            self.E = np.transpose(nip.loadings.fillna(0).to_numpy())  # for small encod_dim nan weights possible
        else:
            pca = PCA(n_components=self.encoding_dim, svd_solver='full')
            pca.fit(adata.X) # X shape: n_samples x n_features
            self.E = pca.components_  # encod_dim x features
        self.E = tf.convert_to_tensor(np.transpose(self.E), dtype=adata.X.dtype)
        
        
LATENT_SPACE_MODELS = {'AE': Encoder_AE, 'PCA': Encoder_PCA}
