import time
import anndata

from .preprocess import preprocess
from .models import Autoencoder_Model
from .results import call_outliers
from .utils import print_func

def outrider(adata,
             encod_dim,  
             prepro_func='none', # preprocessing options
             sf_norm=True, 
             data_trans='none', 
             centering=True,
             noise_factor=0.0,
             latent_space_model='AE', # model parameters
             decoder_model='AE',
             dispersion_model='ML',
             loss_distribution='gaussian',
             optimizer='lbfgs',
             num_cpus=1,
             seed=7,
             iterations=15,     # training options
             convergence=1e-5, 
             initialize=True, 
             verbose=False,
             distribution='gaussian',  # outlier calling options
             fdr_method='fdr_by',
             effect_type='zscore'):
    """
    Runs the OUTRIDER algorithm on the input data.
    :param data: AnnData object
    :param prepro_func: Name of the function that should be applied to the input 
        data before fitting the model. Currently only 'none' (no preprocessing in this case), 
        'log', 'log1p' or 'vst' are implemented.
    :param sf_norm: Boolean value indicating whether size factor normalization should be applied.
    :param data_trans: Name of the function that should be applied to the preprocessed input 
        data to transform for use in the model. Currently only 'none' (no transformation in this case) 
        'log' or 'log1p' are implemented.
    :param centering: Boolean value indication whether the input data should be centered 
        by feature before fitting.
    :param noise_factor: Controls the level of noise added to the input before denoising. 
        Default: 0.0 (no noise added).
    :param encod_dim: Integer value giving the dimension of the latent space.
    :param latent_space_model: Name of the model for fitting the latent space. Currently 
        supported options are 'AE' or 'PCA'.
    :param decoder_model: Name of the model for fitting the decoder. Currently 
        supported options are 'AE' or 'PCA'.
    :param dispersion_model: Name of the model for fitting the dispersion parameters (if any). 
        Currently supported options are 'ML' (maximum likelihood) or 'MoM' (method of moments).
    :param loss_distribution: Name of the distribution that should be used for calculating the model loss. One of 'NB', 'gaussian' or 'log-gaussian'.
    :param optimizer: Name of the optimizer used for fitting the model. Currently only 
        'lbfgs' is implemented.
    :param num_cpus: Number of cores for parallelization.
    :param seed: Seed for random calculations.
    :param iterations: Maximum number of iterations when fitting the model.
    :param convergence: Convergence limit to determine when to stop fitting.
    :param verbose: Boolean value indicating whether additional information should be printed during fitting.
    :param distribution: The distribution for calculating p values. One of 'NB', 'gaussian' or 'log-gaussian'.
    :param fdr_method: name of pvalue adjustment method (from statsmodels.stats.multitest.multipletests).
    :param effect_type: the type of method for effect size calculation. Must be one or a list of 'none', 'fold-change', 'zscores' or 'delta'.
    :return: An AnnData object annotated with model fit results, pvalues, adjusted pvalues and effect sizes.
    """
    
    # check function arguments
    assert isinstance(adata, anndata.AnnData), 'adata must be an AnnData instance'
    assert encod_dim > 1, 'encoding_dim must be >= 2'
    
    # 1. preprocess and prepare data: 
    adata = preprocess(adata, 
                      prepro_func=prepro_func, 
                      sf_norm=sf_norm, 
                      transformation=data_trans,
                      centering=centering,
                      noise_factor=noise_factor)
    
    # 2. build autoencoder model: 
    model = Autoencoder_Model(encoding_dim=encod_dim,
                                     encoder=latent_space_model,
                                     decoder=decoder_model,
                                     dispersion_fit=dispersion_model,
                                     loss_distribution=loss_distribution,
                                     optimizer=optimizer,
                                     num_cpus=num_cpus,
                                     seed=seed,
                                     float_type=adata.X.dtype.name,
                                     verbose=verbose)
    
    # 3. fit model: 
    time_ae_start = time.time()
    print_func.print_time('start model fitting')
    model.fit(adata, initialize=True, iterations=iterations, 
                    convergence=convergence, verbose=verbose)
    print_func.print_time(f'complete model fit time {print_func.get_duration_sec(time.time() - time_ae_start)}')
    
    # E_fit = model.encoder.get_encoder()
    # D_fit, b_fit = model.decoder.get_decoder()
    # disp_fit = model.dispersion_fit.get_dispersions()
    # print(f"E_fit[:3, :5] = {E_fit[:3,:5]}")
    # print(f"D_fit[:3, :5] = {D_fit[:3,:5]}")
    # print(f"b_fit[:5] = {b_fit[:5]}")
    # print(f"disp_fit = {disp_fit}")
    
    # print(f"adata prior get_loss: {adata}")
    print_func.print_time(f'model_fit ended with loss: {model.get_loss(adata)}')
    # adata.write("test/adata_test.h5ad")
    
    # 4. call outliers / get (adjusted) pvalues:
    adata = model.predict(adata)
    # print(f"adata after predict: {adata}")
    adata = call_outliers(adata, 
                        distribution=distribution,
                        fdr_method=fdr_method, 
                        effect_type=effect_type,
                        num_cpus=num_cpus)
        
    return adata 
        
        
        
