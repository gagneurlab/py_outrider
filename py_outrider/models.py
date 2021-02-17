import time
import tensorflow as tf
import numpy as np

from .utils import tf_init
from .utils.loss_list import Loss_list
from .utils import print_func

from .latent_space_models import LATENT_SPACE_MODELS
from .decoder_models import DECODER_MODELS
from .dispersion_models import DISPERSION_MODELS
from .distributions import DISTRIBUTIONS

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
        
        # loss
        self.loss_list = Loss_list()
        self.loss_distribution = loss_distribution
        self.loss_func = loss_distribution.loss
        
        # model params
        self.encoding_dim = encoding_dim
        self.encoder = encoder(encoding_dim=encoding_dim, loss=self.loss_func)
        self.decoder = decoder(loss=self.loss_func)
        self.dispersion_fit = dispersion_fit(loss_distribution)
        
        # train params (only passed to fit func?)
        self.optimizer = optimizer
        self.batch_size = None
        self.n_parallel = num_cpus
        
        # init tf
        tf_init.init_tf_config(num_cpus=num_cpus, verbose=verbose)
        tf_init.init_float_type(float_type=float_type)
        tf_init.init_tf_seed(seed=seed)
        # tf.config.run_functions_eagerly(True)
        
    def predict(self, adata):
        x_latent = self.encoder.encode(adata.X)
        adata.obsm["X_latent"] = x_latent.numpy()
        adata.varm['E'] = self.encoder.get_encoder()
        pred = self.decoder.decode(x_latent)
        adata.layers["X_predicted"] = pred[0].numpy()
        adata.layers["X_predicted_no_trans"] = pred[1].numpy()
        D, adata.varm['bias'] = self.decoder.get_decoder()
        adata.varm['D'] = np.transpose(D)
        if self.loss_distribution.has_dispersion() is True:
            adata.varm["dispersions"] = self.dispersion_fit.get_dispersions()
        return adata
        
    @tf.function
    def predict_internal(self, X, E, D, bias, sf, trans_fun):
        pred = self.decoder._decode(self.encoder._encode(X, E), D, bias, sf, trans_fun)[0]
        disp = self.dispersion_fit.get_dispersions()
        return pred, disp 
        
    def get_loss(self, adata):
        adata = self.predict(adata)
        # print(f'x_true: {adata.layers["X_prepro"]}')
        # print(f'x_pred: {adata.layers["X_predicted"]}')
        # print(f'theta: {adata.varm["dispersions"]}')
        if self.loss_distribution.has_dispersion() is True:
            loss = self.loss_func(adata.layers["X_prepro"], adata.layers["X_predicted"], adata.varm["dispersions"])
        else:
            loss = self.loss_func(adata.layers["X_prepro"], adata.layers["X_predicted"])
        return loss

    def init(self, x, x_true, feature_means, sf, trans_func):
        # init encoder
        print('Initializing the encoder ...')
        self.encoder.init(x)
        new_E = self.encoder.get_encoder()
        
        # init decoder
        print('Initializing the decoder ...')
        self.decoder.init(self.encoder, feature_means, sf, trans_func)
        new_D, new_b = self.decoder.get_decoder()
        
        # init dispersions (if needed)
        if self.loss_distribution.has_dispersion() is True:
            print('Initializing the dispersions ...')
            self.dispersion_fit.init(x_true)
        
        new_disps=self.dispersion_fit.get_dispersions()
        x_pred = self.predict_internal(x, new_E, new_D, new_b, sf, trans_func)[0].numpy()
        init_loss = self.loss_func(x_true=x_true, x_pred=x_pred, dispersions=new_disps)
        self.loss_list.add_loss(init_loss.numpy(),
                                step_name="init_E_D_dispersion", 
                                print_text='Initial - loss:')
        
    def fit(self, adata, initialize=True, iterations=15, convergence=1e-5, 
            verbose=False):
        
        encoder_name = self.encoder.__class__.__name__
        decoder_name = self.decoder.__class__.__name__
        dispersion_name = self.dispersion_fit.__class__.__name__
        
        # extract needed data
        x_in = adata.X
        x_true = adata.layers["X_prepro"]
        feature_means = adata.varm['means'] if "means" in adata.varm.keys() else None
        sf = adata.obsm["sizefactors"]
        trans_func = adata.uns["transform_func"]
        
        if initialize is True:
            self.init(x=x_in, x_true=x_true, feature_means=feature_means, sf=sf, trans_func=trans_func)

        for iter in range(iterations):
            print(f'### ITERATION {iter+1}')
            time_iter_start = time.time()
            
            # iteratively update encoder
            # old_latent = adata.obsm["X_latent"]
            current_dispersions = self.dispersion_fit.get_dispersions()
            # old_E = self.encoder.get_encoder()
            e_loss = self.encoder.fit(x_in=x_in, 
                                    x_true=x_true, 
                                    decoder=self.decoder.copy(),
                                    optimizer=self.optimizer, 
                                    n_parallel=self.n_parallel,
                                    dispersions=current_dispersions)
            new_E = self.encoder.get_encoder()
            # print(f"E changed in {np.sum(old_E != new_E)} positions")
            # e_loss = self.get_loss(adata)
            # new_latent = adata.obsm["X_latent"]
            # print(f"H changed in {np.sum(old_latent != new_latent)} positions")
            # print(f"old_H vs. vs. new_H at [5,10]: {old_latent[5,10]} vs. {new_latent[5,10]}")
            self.loss_list.add_loss(e_loss.numpy(),
                                    step_name=encoder_name, 
                                    print_text=f'{encoder_name} - loss:')
            
            # update decoder
            # old_D, old_b = self.decoder.get_decoder()
            x_latent = self.encoder._encode(x_in, new_E).numpy()
            d_loss = self.decoder.fit(x_latent=x_latent, 
                                    x_true=x_true,
                                    optimizer=self.optimizer, 
                                    n_parallel=self.n_parallel,
                                    dispersions=current_dispersions)
            new_D, new_b = self.decoder.get_decoder()
            # print(f"D changed in {np.sum(old_D != new_D)} positions")
            # print(f"old_D vs. vs. new_D at [5,10]: {old_D[5,10]} vs. {new_D[5,10]}")
            # print(f"b changed in {np.sum(old_D != new_D)} positions")
            # print(f"old_b vs. new_b at [15]: {old_b[15]} vs. {new_b[15]}")
            # d_loss = self.get_loss(adata)
            self.loss_list.add_loss(d_loss.numpy(),
                                    step_name=decoder_name, 
                                    print_text=f'{decoder_name} - loss:')
            
            # update dispersions (if needed)
            steps = 2
            if self.loss_distribution.has_dispersion() is True:
                x_pred = self.predict_internal(x_in, new_E, new_D, new_b, sf, trans_func)[0].numpy()
                dispersion_loss = self.dispersion_fit.fit(x_true=x_true, 
                                                        x_pred=x_pred,
                                                        optimizer='fminbound',
                                                        # optimizer=self.optimizer,
                                                        n_parallel=self.n_parallel)
                # dispersion_loss = self.get_loss(adata)
                self.loss_list.add_loss(dispersion_loss.numpy(),
                                        step_name=dispersion_name, 
                                        print_text=f'{dispersion_name} - loss:')
                steps = steps + 1 
            
            print('duration loop: {}'.format(print_func.get_duration_sec(time.time() - time_iter_start)))
            
            # check convergence after every iteration
            if self.loss_list.check_converged(conv_limit=convergence, 
                                              verbose=verbose,
                                              last_iter=steps):
                print_func.print_time(f'model converged at iteration: {iter+1}')
                break
        
        # E_final = self.encoder.get_encoder()
        # D_final, b_final = self.decoder.get_decoder()
        # disp_final = self.dispersion_fit.get_dispersions()
        # print(f"E_final[:3, :5] = {E_final[:3,:5]}")
        # print(f"D_final[:3, :5] = {D_final[:3,:5]}")
        # print(f"b_final[:5] = {b_final[:5]}")
        # print(f"disp_final = {disp_final}")
    
