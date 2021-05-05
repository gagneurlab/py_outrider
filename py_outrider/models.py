import time
import tensorflow as tf
import numpy as np
import math
import warnings

from .utils import tf_init
from .utils.loss_list import Loss_list
from .utils import print_func
from .preprocess import get_k_most_variable_features

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
                 parallelize_by_feature=False,
                 batch_size=None,
                 nr_latent_space_features=None,
                 num_cpus=1,
                 verbose=False,
                 seed=None,
                 float_type='float64'):

        self.encoder_model_name = encoder
        self.decoder_model_name = decoder
        self.dispersion_model_name = dispersion_fit
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

        # train params
        self.optimizer = optimizer
        self.parallelize_by_feature = parallelize_by_feature
        self.n_parallel = num_cpus
        if batch_size is not None:
            batch_size = int(batch_size)
            if batch_size < 1:
                warnings.warn(f"batch_size={batch_size} <= 0 interpreted "
                              "as None")
                batch_size = None
        self.batch_size = batch_size
        if nr_latent_space_features is not None:
            nr_latent_space_features = int(nr_latent_space_features)
            if nr_latent_space_features < 1:
                warnings.warn(
                    f"nr_latent_space_features={nr_latent_space_features} <= 0"
                    " interpreted as None")
                nr_latent_space_features = None
        self.nr_latent_space_features = nr_latent_space_features

        # init tf
        tf_init.init_tf_config(num_cpus=num_cpus, verbose=verbose)
        tf_init.init_float_type(float_type=float_type)
        tf_init.init_tf_seed(seed=seed)
        # tf.config.run_functions_eagerly(True)

    @staticmethod
    def batch_split(adata, batch_size):
        nr_samples = adata.n_obs
        sample_idx = np.arange(nr_samples)

        if batch_size is None or batch_size >= n_samples:
            return [sample_idx]

        np.random.shuffle(sample_idx)
        nr_batches = math.ceil(nr_samples / batch_size)
        batch_split = list()
        for b in range(nr_batches):
            start_idx = b*batch_size
            end_idx = (b+1)*batch_size
            batch_split.append(sample_idx[start_idx:end_idx])
        return batch_split

    def predict(self, adata):
        # set decoder info for reverse transforming according to adata
        x_na = np.isfinite(adata.layers["X_raw"])
        sf = adata.obsm["sizefactors"]
        self.decoder.sf = sf
        self.decoder.x_na = x_na
        if "covariates_oneh" in adata.uns.keys():
            cov_oneh = adata.uns["covariates_oneh"]
            self.decoder.cov = cov_oneh
        else:
            self.decoder.cov = None

        # encoding
        features = get_k_most_variable_features(adata,
                                                self.nr_latent_space_features)
        x_in = adata.uns["X_AE_input"][:, features]
        x_latent = self.encoder.encode(x_in)
        adata.obsm["X_latent"] = x_latent.numpy()
        if "covariates_oneh" in adata.uns.keys():
            adata.obsm["X_latent_with_cov"] = np.concatenate(
                                                [x_latent.numpy(),
                                                 adata.uns["covariates_oneh"]],
                                                axis=1)
        adata.uns['E'] = self.encoder.get_encoder()

        # decoding
        pred = self.decoder.decode(x_latent)
        adata.layers["X_predicted"] = pred[0].numpy()
        adata.layers["X_predicted_no_trans"] = pred[1].numpy()
        adata.uns['D'], adata.uns['bias'] = self.decoder.get_decoder()

        # dispersions
        if self.loss_distribution.has_dispersion() is True:
            adata.varm["dispersions"] = self.dispersion_fit.get_dispersions()

        return adata

    @tf.function
    def predict_internal(self, X, E, D, bias, sf, trans_fun, x_na, cov):
        pred = self.decoder._decode(self.encoder._encode(X, E), D, bias, sf,
                                    trans_fun, x_na, cov)[0]
        disp = self.dispersion_fit.get_dispersions()
        return pred, disp

    def get_loss(self, adata):
        adata = self.predict(adata)
        if self.loss_distribution.has_dispersion() is True:
            loss = self.loss_func(adata.layers["X_prepro"],
                                  adata.layers["X_predicted"],
                                  adata.varm["dispersions"])
        else:
            loss = self.loss_func(adata.layers["X_prepro"],
                                  adata.layers["X_predicted"])
        return loss

    def init(self, x, x_true, feature_means, sf, trans_func, x_na, cov):
        # init encoder
        print_func.print_time('Initializing the encoder ...')
        self.encoder.init(x)
        new_E = self.encoder.get_encoder()

        # init decoder
        print_func.print_time('Initializing the decoder ...')
        self.decoder.init(self.encoder, x_na, feature_means, sf, trans_func,
                          cov)
        new_D, new_b = self.decoder.get_decoder()

        # init dispersions (if needed)
        if self.loss_distribution.has_dispersion() is True:
            print_func.print_time('Initializing the dispersions ...')
            self.dispersion_fit.init(x_true)

        new_disps = self.dispersion_fit.get_dispersions()
        x_pred = self.predict_internal(x, new_E, new_D, new_b, sf, trans_func,
                                       x_na, cov)[0].numpy()
        init_loss = self.loss_func(x_true=x_true, x_pred=x_pred,
                                   dispersions=new_disps)
        self.loss_list.add_loss(init_loss.numpy(),
                                step_name="init_E_D_dispersion",
                                print_text='Initial - loss:')

    def fit(self, adata, initialize=True, iterations=15, convergence=1e-5,
            verbose=False):

        # get class names for printing
        encoder_name = self.encoder.__class__.__name__
        decoder_name = self.decoder.__class__.__name__
        dispersion_name = self.dispersion_fit.__class__.__name__

        # extract needed data
        x_na = np.isfinite(adata.layers["X_raw"])
        x_in = adata.uns["X_AE_input"]
        x_true = adata.layers["X_prepro"]
        if "means" in adata.varm.keys():
            feature_means = adata.varm['means']
        else:
            feature_means = None
        sf = adata.obsm["sizefactors"]
        trans_func = adata.uns["transform_func"]
        if "covariates_oneh" in adata.uns.keys():
            cov_oneh = adata.uns["covariates_oneh"]
        else:
            cov_oneh = None

        if initialize:
            self.init(x=x_in, x_true=x_true, feature_means=feature_means,
                      sf=sf, trans_func=trans_func, x_na=x_na, cov=cov_oneh)
        else:
            if "E" in adata.uns.keys():
                self.encoder.E = adata.uns["E"]
                self.encoder.encoding_dim = adata.uns["E"].shape[1]
                self.encoder.loss = self.loss_func
            else:
                raise ValueError("No encoding matrix E found in adata.uns.",
                                 "This is required when initialize=False")
            if "D" in adata.uns.keys():
                self.decoder.D = tf.convert_to_tensor(adata.uns["D"], dtype=x_in.dtype)
                self.decoder.bias = tf.convert_to_tensor(adata.uns["bias"])
                self.decoder.loss = self.loss_func
                self.decoder.cov = cov_oneh
                self.decoder.sf = sf
                self.decoder.rev_trans = trans_func
            else:
                raise ValueError("No decoding matrix D found in adata.uns.",
                                 "This is required when initialize=False")
            if "dispersions" in adata.varm.keys():
                self.dispersion_fit.dispersions = tf.convert_to_tensor(adata.varm["dispersions"])

        # subset to most variable features for encoder fitting if requested
        features = get_k_most_variable_features(adata,
                                                self.nr_latent_space_features)
        self.encoder = self.encoder.subset(features)

        # iteratively update encoder, decoder and dispersions
        for iter in range(iterations):
            print(f'### ITERATION {iter+1}')
            time_iter_start = time.time()

            # split adata into batches and update after each batch
            batches = self.batch_split(adata, self.batch_size)
            for batch_id in range(len(batches)):
                # subset matrices to samples in batch
                b = batches[batch_id]
                x_in = adata.uns["X_AE_input"][b, :][:, features]
                x_true = adata[b, :].layers["X_prepro"]
                x_na = np.isfinite(adata[b, :].layers["X_raw"])
                sf = adata[b, :].obsm["sizefactors"]
                self.decoder.sf = sf
                self.decoder.x_na = x_na
                if "covariates_oneh" in adata.uns.keys():
                    cov_oneh = adata.uns["covariates_oneh"][b, :]
                    self.decoder.cov = cov_oneh

                # update encoder
                # print(f'### # Fitting the latent space ...')
                print(f'###\tBATCH {batch_id+1}/{len(batches)}: ', end='')
                current_dispersions = self.dispersion_fit.get_dispersions()
                e_loss = self.encoder.fit(x_in=x_in,
                                          x_true=x_true,
                                          decoder=self.decoder.copy(),
                                          optimizer=self.optimizer,
                                          n_parallel=self.n_parallel,
                                          dispersions=current_dispersions)
                new_E = self.encoder.get_encoder()
                self.loss_list.add_loss(e_loss.numpy(),
                                        step_name=encoder_name,
                                        print_text=f'{encoder_name} - loss:')

                # update decoder
                # print(f'### # Fitting the decoder ...')
                print(f'###\tBATCH {batch_id+1}/{len(batches)}: ', end='')
                x_latent = self.encoder._encode(x_in, new_E).numpy()
                d_loss = self.decoder.fit(
                            x_latent=x_latent,
                            x_true=x_true,
                            optimizer=self.optimizer,
                            parallelize_by_feature=self.parallelize_by_feature,
                            n_parallel=self.n_parallel,
                            dispersions=current_dispersions)
                new_D, new_b = self.decoder.get_decoder()
                self.loss_list.add_loss(d_loss.numpy(),
                                        step_name=decoder_name,
                                        print_text=f'{decoder_name} - loss:')

                # update dispersions (if needed)
                steps = 2
                if self.loss_distribution.has_dispersion() is True:
                    # print(f'### # Fitting the dispersion parameters ...')
                    print(f'###\tBATCH {batch_id+1}/{len(batches)}: ', end='')
                    x_pred = self.predict_internal(x_in, new_E, new_D, new_b,
                                                   sf, trans_func, x_na,
                                                   cov_oneh)[0].numpy()
                    dispersion_loss = self.dispersion_fit.fit(
                                                    x_true=x_true,
                                                    x_pred=x_pred,
                                                    optimizer='fminbound',
                                                    n_parallel=self.n_parallel)
                    self.loss_list.add_loss(dispersion_loss.numpy(),
                                            step_name=dispersion_name,
                                            print_text=(
                                                f'{dispersion_name} - loss:'))
                    steps = steps + 1

            print('duration loop: {}'.format(print_func.get_duration_sec(
                                               time.time() - time_iter_start)))

            # check convergence after every iteration
            if self.loss_list.check_converged(conv_limit=convergence,
                                              verbose=verbose,
                                              last_iter=steps*len(batches)):
                print_func.print_time((
                    f'model converged at iteration: {iter+1}'))
                break
