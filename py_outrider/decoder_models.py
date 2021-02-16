import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class Decoder_AE():

    def __init__(self):
        self.D = None
        self.bias = None
        
    def init(self, adata, encoder):
        self.D = tf.transpose(encoder.E)
        if "means" in adata.varm.keys():
            self.bias = adata.varm['means']
        else:
            self.bias = np.zeros(adata.n_vars)
        self.bias = tf.convert_to_tensor(self.bias)
        
    
    def decode(self, adata):
        prediction = tf.matmul(tf.convert_to_tensor(adata.obsm['X_latent']), self.D)
        prediction = tf.gather(prediction, tf.range(self.bias.shape[0]), axis=1)
        prediction = tf.math.add(prediction, self.bias)
        adata.layers['X_predicted'] = prediction
        adata.varm['D'] = tf.transpose(self.D)
        adata.varm['bias'] = self.bias
        return adata
        
    @tf.function
    def loss_func_d(self, b_and_D, adata, loss_func):
        new_dec = Decoder_AE()
        b_and_D_out = tf.reshape(b_and_D, [self.D.shape[0] + 1, self.D.shape[1]])
        new_dec.D = tf.convert_to_tensor(b_and_D_out[1:, :], dtype=adata.X.dtype.name)
        new_dec.bias = tf.convert_to_tensor(b_and_D_out[0, :], dtype=adata.X.dtype.name)
        return loss_func(decoder=new_dec, adata=adata)
        
    @tf.function
    def fit(self, adata, loss_func, optimizer, n_parallel):
        if optimizer == "lbfgs":
            b_and_D = tf.concat([tf.expand_dims(self.bias, 0), self.D], axis=0)
            b_and_D = tf.reshape(b_and_D, [-1])  ### flatten
    
            def lbfgs_input(b_and_D):
                loss = self.loss_func_d(b_and_D=b_and_D, adata=adata, loss_func=loss_func)
                gradients = tf.gradients(loss, b_and_D)[0]  ## works but runtime check, eager faster ??
                return loss, tf.clip_by_value(tf.reshape(gradients, [-1]), -100., 100.)
    
            optim = tfp.optimizer.lbfgs_minimize(lbfgs_input, 
                                                 initial_position=b_and_D, 
                                                 tolerance=1e-8, 
                                                 max_iterations=100, #300, #100,
                                                 num_correction_pairs=10, 
                                                 parallel_iterations = n_parallel)
            b_and_D_out = tf.reshape(optim.position, [self.D.shape[0] + 1, self.D.shape[1]])
            self.D = tf.convert_to_tensor(b_and_D_out[1:, :], dtype=adata.X.dtype.name)
            self.b = tf.convert_to_tensor(b_and_D_out[0, :], dtype=adata.X.dtype.name)
            
        

class Decoder_PCA():

    def __init__(self):
        self.D = None
        self.bias = None
        
    def init(self, adata, encoder):
        self.D = tf.transpose(encoder.E)
        if "means" in adata.varm.keys():
            self.bias = adata.varm['means']
        else:
            self.bias = np.zeros(adata.n_vars)
        self.bias = tf.convert_to_tensor(self.bias)
        
    
    def decode(self, adata):
        prediction = tf.matmul(tf.convert_to_tensor(adata.obsm['X_latent']), self.D)
        prediction = tf.gather(prediction, tf.range(self.bias.shape[0]), axis=1)
        prediction = tf.math.add(prediction, self.bias)
        adata.layers['X_predicted'] = prediction
        adata.varm['D'] = tf.transpose(self.D)
        adata.varm['bias'] = self.bias
        return adata
        
    def fit(self, adata, loss_func, optimizer, n_parallel):
        pass     
    

DECODER_MODELS = {'AE': Decoder_AE, 'PCA': Decoder_PCA}

