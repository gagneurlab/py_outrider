import argparse
from pathlib import Path
import shutil
import os


def parse_args(args_input):
    parser = argparse.ArgumentParser(description='run outrider')
    # input / output params
    parser.add_argument('-in','--input', type=str, help='Path to a file containing the input data matrix, like a gene count table, format: rows:samples x columns:genes, must be comma separated, needs 1. row and 1. column to have names.' ) #, required=True )
    parser.add_argument('-sa','--sample_anno', default=None, type=str, help='Path to sample annotation file, must be comma separated, automatically selects sampleID column as the one which has input_file rownames in it.' )
    parser.add_argument('-o','--output', type=str, default=None, help='Folder to save results (xarray object, plots, result list), creates new folder if it does not exist, otherwise creates new folder within to save xarray object.')
    parser.add_argument('-ot','--output_type', type=str, default='h5ad', choices=['h5ad', 'csv', 'zarr'], help='File type of the output (h5ad, csv or zarr).')
    # parser.add_argument('-op','--output_plots', type=bool, default=True, help='Outputs a collection of useful plots.')
    # parser.add_argument('-ol','--output_list', type=bool, default=False, help='Outputs result in form of a long list.')

    # outrider params
    parser.add_argument('-p','--profile', default='outrider', choices=['outrider', 'protrider', 'pca'], help='Choose which pre-defined implementation should be used.')
    parser.add_argument('-q','--encod_dim', type=int, default=None, help='Number of encoding dimensions, if None: runs hyperparameter optimization to determine best encoding dimension.')#, required=True)
    parser.add_argument('-cov','--covariates', default=None, nargs='*', help='List of covariate names in sample_anno, can either be columns filled with 0 and 1, multiple numbers or strings (will be automatically converted to one-hot encoded matrix for training).' )
    parser.add_argument('-i','--iterations', type=check_positive_int, default=None, help='[predefined in profile] Number of maximal training iterations.')
    parser.add_argument('-fl','--float_type', default=None, choices=['float32', 'float64'], help='[predefined in profile] Which float type should be used, highly advised to keep float64 which may take longer, but accuracy is strongly impaired by float32.')
    parser.add_argument('-cpu','--num_cpus', type=check_positive_int, default=None, help='Number of cpus used. Default is 1.')
    parser.add_argument('-v','--verbose', type=bool, default=None, help='Print additional output info during training. Default is False.')
    parser.add_argument('-s','--seed', type=int, default=None, help='Seed used for training, negative values (e.g. -1) -> no seed set. Default is 7.')
    parser.add_argument('-dis','--distribution', default=None, choices=['NB', 'gaussian'], help='[predefined in profile] distribution assumed for the measurement data.')
    parser.add_argument('-ld','--loss_distribution', default=None, choices=['NB', 'gaussian'], help='[predefined in profile] loss distribution used for training.')
    parser.add_argument('-pre','--prepro_func', default=None, choices=['none', 'log'], help='[predefined in profile] preprocess data before input.')
    parser.add_argument('-sf', '--sf_norm', default=None, type=bool, help='[predefined in profile] Boolean value indicating whether sizefactor normalization should be performed.')
    parser.add_argument('-c', '--centering', default=None, type=bool, help='[predefined in profile] Boolean value indicating whether input should be centered before model fitting.')
    parser.add_argument('-dt','--data_trans', default=None, choices=['sf', 'log', 'none'], help='[predefined in profile] change transformation scheme for measurement data during training.')
    parser.add_argument('-nf','--noise_factor', default=None, type=float, help='[predefined in profile] factor which defines the amount of noise applied for a denoising autoencoder model.')
    parser.add_argument('--latent_space_model', default=None, choices=['AE', 'PCA'], help='[predefined in profile] Sets the model for fitting the latent space.')
    parser.add_argument('--decoder_model', default=None, choices=['AE', 'PCA'], help='[predefined in profile] Sets the model for fitting the decoder.')
    parser.add_argument('--dispersion_model', default=None, choices=['ML', 'MoM'], help='[predefined in profile] Sets the model for fitting the dispersion parameters. Either ML for maximum likelihood fit or MoM for methods of moments.')
    parser.add_argument('--optimizer', default=None, choices=['lbfgs'], help='[predefined in profile] Sets the optimizer for model fitting.')
    parser.add_argument('--convergence', default=None, type=check_positive, help='Sets the convergence limit. Default value is 1e-5.')
    parser.add_argument('--effect_type', default=None, type=str, choices=['none', 'zscores', 'fold_change', 'delta'], help='[predefined in profile] Chooses the type of effect size that is calculated.')
    parser.add_argument('--fdr_method', default=None, type=str, help='Sets the fdr adjustment method. Must be one of the methods from statsmodels.stats.multitest.multipletests. Defaults to fdr_by.')
    # parser.add_argument('--batch_size', type=int, default=None, help='batch_size used for stochastic training. Default is to not train stochastically.')
    # parser.add_argument('--parallelize_D', action='store_true', dest='parallelize_D', help='If this flag is given, parallelizes fitting of decoder per feature. Default behavior depends on sample size.')
    # parser.add_argument('--no_parallelize_D', action='store_false', dest='parallelize_D', help='If this flag is given, decoder fit will not be parallelized per feature. Default behavior depends on sample size.')
    # parser.set_defaults(parallelize_D=None)
    # parser.add_argument('--X_is_outlier', type=str, default=None, help='Path to a file (same shape as input) with values of 0|1 for injected outliers, automatically performs precision-recall on in.')

    args = parser.parse_args(args_input)
    return vars(args)

def check_positive_int(value):
    ivalue = int(value)
    return check_positive(ivalue)

def check_positive(value):
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive value" )
    return value


def extract_outrider_args(args):
    
    if 'profile' in args:
        outrider_args = construct_profile_args(args['profile'])
    
    # overwrite values from profile if requested
    outrider_params = ['prepro_func', 'sf_norm', 'data_trans', 'centering',
                       'noise_factor', 'encod_dim', 'latent_space_model',
                       'decoder_model', 'dispersion_model', 'loss_distribution',
                       'optimizer', 'num_cpus', 'seed', 'iterations', 
                       'convergence', 'verbose', 'distribution',
                       'fdr_method','effect_type']
    for param in outrider_params:
        if args[param] is not None:
            outrider_args[param] = args[param]
    
    return outrider_args
    
def construct_profile_args(profile):
    assert profile in ('outrider', 'protrider', 'pca'), 'Unknown profile specified.' 
    
    outrider_args = dict({'prepro_func': 'none', 
                          'sf_norm': True,
                          'data_trans': 'none',
                          'centering': True,
                          'noise_factor': 0.0,
                          'encod_dim': None,
                          'latent_space_model': 'AE',
                          'decoder_model': 'AE',
                          'dispersion_model': 'ML',
                          'loss_distribution': 'gaussian',
                          'optimizer': 'lbfgs',
                          'num_cpus': 1,
                          'seed': 7,
                          'iterations': 15,
                          'convergence': 1e-5,
                          'verbose': False,
                          'distribution': 'gaussian',
                          'fdr_method': 'fdr_by',
                          'effect_type': 'none'})
    
    if profile == 'outrider':
        outrider_args['data_trans'] = 'log1p'
        outrider_args['loss_distribution'] = 'NB'
        outrider_args['distribution'] = 'NB'
        outrider_args['effect_type'] = 'fold_change'
    elif profile == 'protrider':
        outrider_args['prepro_func'] = 'log1p'
        outrider_args['effect_type'] = 'zscores'
    elif profile == 'pca':
        outrider_args['effect_type'] = 'zscores'
        outrider_args['latent_space_model'] = 'PCA'
        outrider_args['decoder_model'] = 'PCA'
        
    return outrider_args

class Check_parser():

    def __init__(self, args_input):

        ### modify arguments and transform to path obj
        self.args_mod = args_input.copy()
        self.args_mod['input'] = self.check_input_file(args_input['input'])
        self.args_mod['sample_anno'] = self.check_sample_anno(args_input['sample_anno'])
        self.args_mod['encod_dim'] = self.check_encod_dim(args_input['encod_dim'])
        self.args_mod['output'] = self.check_output(args_input['input'], args_input['output'])
        # self.args_mod['X_is_outlier'] = self.check_X_is_outlier(args_input['X_is_outlier'])

### TODO CHECK COV USED IS LIST




    def get_path(self, file_path, name):
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f'{name} does not exists: {file_path}')
        else:
            return path


    def check_input_file(self, input_file):
        if input_file is None:
            raise ValueError("input file path not specified")
        else:
            return self.get_path(input_file, "input file")


    def check_sample_anno(self, sample_anno):
        if sample_anno is None:
            return None
        else:
            return self.get_path(sample_anno, "sample_anno file")


    def check_encod_dim(self, encod_dim):
        if encod_dim is not None and encod_dim < 1:
                raise ValueError(f'encod_dim must be >0: {encod_dim}')
        return encod_dim


    def check_output(self, input_file, output):
        if output is None:
            output = Path(input_file).parent
        else:
            output = Path(output)

        if output.exists() and not output.is_file():
            if not os.listdir(output):  # if empty
                return str(output.resolve())
            else:
                output = output / "outrider_result.zarr"
                if output.exists():
                    shutil.rmtree(output)
                return str(output.resolve())
        else:
            output.mkdir(parents=True, exist_ok=True)
            return str(output.resolve())



    # def check_X_is_outlier(self, X_is_outlier):
    #     if X_is_outlier is None:
    #         return None
    #     else:
    #         return self.get_path(X_is_outlier, "X_is_outlier")


