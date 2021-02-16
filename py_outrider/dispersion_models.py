import tensorflow as tf
import tensorflow_probability as tfp

# Maximum Likelihood fit of dispersion parameters
class Dispersions_ML():

    def __init__(self, distribution):
        self.dispersion = None
        self.distribution = distribution
        
    def get_dispersions(self):
        return self.dispersions
        
    def init(self, adata):
        mom = Dispersions_MoM(self.distribution)
        mom.fit(adata, loss_func=None, optimizer=None, n_parallel=None)
        self.dispersions = mom.dispersions
    
    @tf.function
    def loss_func_disp(self, dispersions, adata, loss_func):
        new_disp = Dispersions_ML(self.distribution)
        new_disp.dispersions = dispersions
        return loss_func(dispersion_fit=new_disp, adata=adata)
    
    @tf.function
    def fit(self, adata, loss_func, optimizer, n_parallel):
        if optimizer == "lbfgs":
            print('### # Fitting the dispersions with lbfgs ...')
            
            def lbfgs_input(disp):
                loss = self.loss_func_disp(dispersions=disp, adata=adata, loss_func=loss_func)
                gradients = tf.gradients(loss, disp)[0]
                return loss, tf.clip_by_value(gradients, -100., 100.)
            
            optim = tfp.optimizer.lbfgs_minimize(lbfgs_input, 
                                                 initial_position=self.dispersions, 
                                                 tolerance=1e-8, 
                                                 max_iterations=100, #500, #150,
                                                 num_correction_pairs=10, 
                                                 parallel_iterations=n_parallel)
            print(optim.position)
            self.dispersions = optim.position
            # print("Finish dispersion lbfgs fit ...")
        
        print("Finish dispersion lbfgs fit ...")
        return None
    
# Method of moment fit for dispersions
class Dispersions_MoM():

    def __init__(self, distribution):
        self.dispersion = None
        self.distribution = distribution
    
    def get_dispersions(self):
        return self.dispersions    
    
    def init(self, adata):
        self.fit(adata, loss_func=None, optimizer=None, n_parallel=None)
        
    def fit(self, adata, loss_func, optimizer, n_parallel):
        # method of moments
        self.dispersions = self.distribution.mom(adata.layers["X_prepro"])
        self.dispersions = tf.convert_to_tensor(self.dispersions)
        
        
DISPERSION_MODELS = {'ML': Dispersions_ML, 'MoM': Dispersions_MoM}

