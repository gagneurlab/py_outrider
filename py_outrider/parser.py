import argparse
from pathlib import Path


def parse_args(args_input):
    parser = argparse.ArgumentParser(
        description='Run py_outrider to detect aberrant events in omics data.')
    # input / output params
    parser.add_argument(
        '-in', '--input', type=str,
        help='Path to a file containing the input data matrix, like a gene '
             'count table. Can either be a csv file with format: rows:samples '
             'x columns:genes and 1. row and 1. column having feature and '
             'sample names, respectively; or a .h5ad file containing an '
             'existing AnnData object .')
    parser.add_argument(
        '-sa', '--sample_anno', default=None, type=str,
        help='Path to sample annotation file, must be comma separated, '
             'automatically selects sampleID column as the one which has '
             'input_file rownames in it.')
    parser.add_argument(
        '-o', '--output', type=str, default=None,
        help='Filename to save results (adata object).')
    parser.add_argument(
        '-ot', '--output_type', type=str, default='h5ad',
        choices=['h5ad', 'csv', 'zarr'],
        help='File type of the output (h5ad, csv or zarr). Default: h5ad')
    parser.add_argument(
        '-or', '--output_res_table', type=str, default=None,
        help='Outputs results table as csv to the specifed file.')

    # outrider params
    parser.add_argument(
        '-p', '--profile', default='outrider',
        choices=['outrider', 'protrider', 'pca'],
        help='Choose which pre-defined implementation should be used. '
             'Default profile: outrider')
    parser.add_argument(
        '-q', '--encod_dim', type=int, default=None,
        help='Number of encoding dimensions. If not specified, runs '
             'hyperparameter optimization to determine best encoding '
             'dimension.')
    parser.add_argument(
        '-cov', '--covariates', default=None, nargs='*',
        help='List of covariate names in sample_anno, can either be columns '
        'filled with 0 and 1, multiple numbers or strings (will be '
        'automatically converted to one-hot encoded matrix for training).')
    parser.add_argument(
        '-i', '--iterations', type=check_positive_or_zero_int, default=None,
        help='[predefined in profile] Number of maximal training iterations. '
             'Default: 15')
    parser.add_argument(
        '-fl', '--float_type', default=None, choices=['float32', 'float64'],
        help='[predefined in profile] Which float type should be used, highly '
             'advised to keep float64 which may take longer, but accuracy is '
             'strongly impaired by float32. Default: float64')
    parser.add_argument(
        '-cpu', '--num_cpus', type=check_positive_int, default=None,
        help='Number of cpus used. Default: 1')
    parser.add_argument(
        '-v', '--verbose', type=bool, default=None,
        help='Set this flag to enable printing of additional information '
             'during model fitting.')
    parser.add_argument(
        '-s', '--seed', type=int, default=None,
        help='Seed used for training, negative values (e.g. -1) -> no seed '
             'set. Default: 7')
    parser.add_argument(
        '-dis', '--distribution', default=None,
        choices=['NB', 'gaussian', 'log-gaussian'],
        help='[predefined in profile] distribution assumed for the '
             'measurement data.')
    parser.add_argument(
        '-ld', '--loss_distribution', default=None,
        choices=['NB', 'gaussian', 'log-gaussian'],
        help='[predefined in profile] loss distribution used for training.')
    parser.add_argument(
        '-pre', '--prepro_func', default=None,
        choices=['none', 'log', 'log2', 'log1p'],
        help='[predefined in profile] preprocessing function applied to '
             'input data. Distribution should be applicable for the '
             'preprocessed data.')
    parser.add_argument(
        '-sf', '--sf_norm', default=None, type=str2bool,
        help='[predefined in profile] Boolean value indicating whether '
             'sizefactor normalization should be performed.')
    parser.add_argument(
        '-c', '--centering', default=None, type=str2bool,
        help='[predefined in profile] Boolean value indicating whether input '
             'should be centered before model fitting.')
    parser.add_argument(
        '-dt', '--data_trans', default=None, choices=['log1p', 'log', 'none'],
        help='[predefined in profile] transformation function applied to '
             'preprocessed input data to create input for AE model.')
    parser.add_argument(
        '-nf', '--noise_factor', default=None, type=float,
        help='[predefined in profile] factor which defines the amount of '
             'noise applied for a denoising autoencoder model.')
    parser.add_argument(
        '--latent_space_model', default=None, choices=['AE', 'PCA'],
        help='[predefined in profile] Sets the model for fitting the latent '
             'space.')
    parser.add_argument(
        '--decoder_model', default=None, choices=['AE', 'PCA'],
        help='[predefined in profile] Sets the model for fitting the decoder.')
    parser.add_argument(
        '--dispersion_model', default=None, choices=['ML', 'MoM'],
        help='[predefined in profile] Sets the model for fitting the '
             'dispersion parameters. Either ML for maximum likelihood fit or '
             'MoM for methods of moments. Has no effect for loss distributions'
             ' that do not have dispersion parameters.')
    parser.add_argument(
        '--optimizer', default=None, choices=['lbfgs'],
        help='[predefined in profile] Sets the optimizer for model fitting. '
             'Currently only L-BFGS is implemented.')
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help='batch_size used for model fitting. Default is to use all '
             'samples.')
    parser.add_argument(
        '--nr_latent_space_features', type=int, default=None,
        help='Limits the number of features used for fitting the latent '
             'space to the specified number k. The k most variable features '
             'will be selected. Default is to use all features.')
    parser.add_argument(
        '--parallelize_D', action='store_true',
        dest='parallelize_decoder_by_feature',
        help='If this flag is set, parallelizes fitting of decoder per '
             'feature. Default: True (do parallelize by feature).')
    parser.add_argument(
        '--no_parallelize_D', action='store_false',
        dest='parallelize_decoder_by_feature',
        help='If this flag is set, decoder fit will not be parallelized per '
             'feature. Default: do parallelize by feature.')
    parser.add_argument(
        '--convergence', default=None, type=check_positive_float,
        help='Sets the convergence limit. Default value is 1e-5.')
    parser.add_argument(
        '--effect_type', default=None, nargs="*",
        choices=['none', 'zscores', 'fold_change', 'delta'],
        help='[predefined in profile] Chooses the type of effect size that is '
             'calculated. Specifying multiple options is possible.')
    parser.add_argument(
        '--fdr_method', default=None, type=str,
        help='Sets the fdr adjustment method. Must be one of the methods '
             'from statsmodels.stats.multitest.multipletests. '
             'Defaults to fdr_by.')
    parser.set_defaults(parallelize_decoder_by_feature=None)

    # hyper par opt parameters
    parser.add_argument(
        '--max_iter_hyper', default=15, type=check_positive_or_zero_int,
        help='Number of maximial training iterations during hyper parameter '
             'optimization. Default: 15')
    parser.add_argument(
        '--convergence_hyper', default=1e-5, type=check_positive_float,
        help='Convergence limit used during hyper parameter optimization. '
             'Default: 1e-5')

    args = parser.parse_args(args_input)
    return vars(args)


def check_positive_int(value):
    ivalue = int(value)
    return check_positive(ivalue)


def check_positive_float(value):
    fvalue = float(value)
    return check_positive(fvalue)


def check_positive(value):
    if value <= 0:
        raise argparse.ArgumentTypeError(
            f"{value} is an invalid positive value")
    return value


def check_positive_or_zero_int(value):
    ivalue = int(value)
    return check_positive_or_zero(ivalue)


def check_positive_or_zero(value):
    if value < 0:
        raise argparse.ArgumentTypeError(
            f"{value} is an invalid positive or zero value")
    return value


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def extract_outrider_args(args):

    if 'profile' in args:
        outrider_args = construct_profile_args(args['profile'])

    # overwrite values from profile if requested
    outrider_params = ['prepro_func', 'sf_norm', 'data_trans', 'centering',
                       'noise_factor', 'encod_dim', 'latent_space_model',
                       'decoder_model', 'dispersion_model',
                       'loss_distribution', 'nr_latent_space_features',
                       'covariates', 'optimizer', 'batch_size', 'num_cpus',
                       'parallelize_decoder_by_feature', 'seed', 'iterations',
                       'convergence', 'verbose', 'distribution', 'fdr_method',
                       'effect_type', 'float_type']
    for param in outrider_params:
        if args[param] is not None:
            outrider_args[param] = args[param]

    return outrider_args


def construct_profile_args(profile):
    assert profile in ('outrider', 'protrider', 'pca'), (
        'Unknown profile specified.')

    outrider_args = dict({'prepro_func': 'none',
                          'sf_norm': True,
                          'data_trans': 'none',
                          'centering': True,
                          'noise_factor': 0.0,
                          'encod_dim': None,
                          'latent_space_model': 'AE',
                          'decoder_model': 'AE',
                          'dispersion_model': 'ML',  # ignored if not NB
                          'loss_distribution': 'gaussian',
                          'covariates': None,
                          'optimizer': 'lbfgs',
                          'parallelize_decoder_by_feature': True,
                          'batch_size': None,
                          'nr_latent_space_features': None,
                          'num_cpus': 1,
                          'float_type': "float64",
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
        outrider_args['prepro_func'] = 'log2'
        outrider_args['noise_factor'] = 0.5
        outrider_args['effect_type'] = ['zscores', 'fold_change', 'delta']
    elif profile == 'pca':
        outrider_args['effect_type'] = ['zscores', 'delta']
        outrider_args['latent_space_model'] = 'PCA'
        outrider_args['decoder_model'] = 'PCA'
        outrider_args['sf_norm'] = False

    return outrider_args


class Check_parser():

    def __init__(self, args_input):

        # modify arguments and transform to path obj
        self.args_mod = args_input.copy()
        self.args_mod['input'] = self.check_input_file(args_input['input'])
        self.args_mod['sample_anno'] = self.check_sample_anno(
                                                args_input['sample_anno'])
        self.args_mod['encod_dim'] = self.check_encod_dim(
                                                args_input['encod_dim'])
        self.args_mod['output'] = self.check_output(args_input['input'],
                                                    args_input['output'])

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
        return str(output.resolve())
