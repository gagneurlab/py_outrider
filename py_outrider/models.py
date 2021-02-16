import time
import tensorflow as tf

from .utils import tf_init
from .utils.loss_list import Loss_list
from .utils import print_func

from .latent_space_models import LATENT_SPACE_MODELS
from .decoder_models import DECODER_MODELS
from .dispersion_models import DISPERSION_MODELS
from .distributions import DISTRIBUTIONS

from .preprocess import reverse_transform

class Autoencoder_Model():

    def __init__(self, 
                 encoding_dim,
                 encoder='AE', 
                 decoder='AE', 
                 dispersion_fit='ML', 
                 loss_distribution='gaussian',
                 optimizer='lbfgs', 
                 num_cpus=1, 
                 verbose=False,
                 seed=None,
                 float_type='float64'):
                     
        encoder = LATENT_SPACE_MODELS[encoder]
        decoder = DECODER_MODELS[decoder]
        dispersion_fit = DISPERSION_MODELS[dispersion_fit]
        loss_distribution = DISTRIBUTIONS[loss_distribution]
        
        # model params
        self.encoding_dim = encoding_dim
        self.encoder = encoder(encoding_dim)
        self.decoder = decoder()
        self.dispersion_fit = dispersion_fit(loss_distribution)
        
        # loss
        self.loss_list = Loss_list()
        self.loss_distribution = loss_distribution
        self.loss_func = loss_distribution.loss
        
        # train params (only passed to fit func?)
        self.optimizer = optimizer
        self.batch_size = None
        self.n_parallel = num_cpus
        
        # init tf
        tf_init.init_tf_config(num_cpus=num_cpus, verbose=verbose)
        tf_init.init_float_type(float_type=float_type)
        tf_init.init_tf_seed(seed=seed)
        tf.config.run_functions_eagerly(True)
        
    def predict(self, adata, encoder=None, decoder=None, dispersion_fit=None):
        if encoder is None:
            encoder = self.encoder
        if decoder is None:
            decoder = self.decoder
        if dispersion_fit is None: 
            dispersion_fit=self.dispersion_fit
            
        adata = decoder.decode(encoder.encode(adata))
        adata = reverse_transform(adata)
        adata.varm["dispersions"] = dispersion_fit.get_dispersions()
        return adata
        
    @tf.function
    def get_loss(self, adata, encoder=None, decoder=None, dispersion_fit=None):
        if encoder is None:
            encoder = self.encoder
        if decoder is None:
            decoder = self.decoder
        if dispersion_fit is None: 
            dispersion_fit=self.dispersion_fit
            
        adata = self.predict(adata, encoder=encoder, decoder=decoder, dispersion_fit=dispersion_fit)
        if self.loss_distribution.has_dispersion() is True:
            loss = self.loss_func(adata.layers["X_prepro"], adata.layers["X_predicted"], adata.varm["dispersions"])
        else:
            loss = self.loss_func(adata.layers["X_prepro"], adata.layers["X_predicted"])
        return loss

    @tf.function            
    def loss_func_encoder(self, encoder, adata):
        return self.get_loss(adata=adata, encoder=encoder)
        
    @tf.function
    def loss_func_decoder(self, decoder, adata):
        return self.get_loss(adata=adata, decoder=decoder)
    
    @tf.function
    def loss_func_dispersions(self, dispersion_fit, adata):
        return self.get_loss(adata=adata, dispersion_fit=dispersion_fit)
        
    def init(self, adata):
        # init encoder
        print('Initializing the encoder ...')
        self.encoder.init(adata)
        
        # init decoder
        print('Initializing the decoder ...')
        self.decoder.init(adata, self.encoder)
        
        # init dispersions (if needed)
        if self.loss_distribution.has_dispersion() is True:
            print('Initializing the dispersions ...')
            self.dispersion_fit.init(adata)
        
        # get initial loss
        self.loss_list.add_loss(self.get_loss(adata), 
                                step_name="init_E_D_dispersion", 
                                print_text='Initial - loss:')
        
    def fit(self, adata, initialize=True, iterations=15, convergence=1e-5, 
            verbose=False):
        
        encoder_name = self.encoder.__class__.__name__
        decoder_name = self.decoder.__class__.__name__
        dispersion_name = self.dispersion_fit.__class__.__name__
        
        if initialize is True:
            self.init(adata)

        for iter in range(iterations):
            print(f'### ITERATION {iter+1}')
            time_iter_start = time.time()
            
            # iteratively update encoder
            self.encoder.fit(adata=adata, 
                             loss_func=self.loss_func_encoder, 
                             optimizer=self.optimizer, 
                             n_parallel=self.n_parallel)
            self.loss_list.add_loss(self.get_loss(adata), 
                                    step_name=encoder_name, 
                                    print_text=f'{encoder_name} - loss:')
            
            # update decoder
            self.decoder.fit(adata=adata, 
                             loss_func=self.loss_func_decoder, 
                             optimizer=self.optimizer, 
                             n_parallel=self.n_parallel)
            self.loss_list.add_loss(self.get_loss(adata), 
                                    step_name=decoder_name, 
                                    print_text=f'{decoder_name} - loss:')
            
            # update dispersions (if needed)
            steps = 2
            if self.loss_distribution.has_dispersion() is True:
                self.dispersion_fit.fit(adata=adata, 
                                        loss_func=self.loss_func_dispersions, 
                                        optimizer=self.optimizer, 
                                        n_parallel=self.n_parallel)
                print("Adding to loss list ...")
                self.loss_list.add_loss(self.get_loss(adata), 
                                        step_name=dispersion_name, 
                                        print_text=f'{dispersion_name} - loss:')
                print("Added to loss list ...")
                steps = steps + 1 
            
            print('duration loop: {}'.format(print_func.get_duration_sec(time.time() - time_iter_start)))
            
            # check convergence after every iteration
            if self.loss_list.check_converged(conv_limit=convergence, 
                                              verbose=verbose,
                                              last_iter=steps):
                print_func.print_time(f'model converged at iteration: {iter+1}')
                break
            
    
