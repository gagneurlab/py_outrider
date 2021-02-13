import argparse
from pathlib import Path
import shutil
import os


def parse_args(args_input):
    parser = argparse.ArgumentParser(description='run outrider')
    parser.add_argument('-in','--input', type=str, help='Path to a file containing the input data matrix, like a gene count table, format: rows:samples x columns:genes, must be comma separated, needs 1. row and 1. column to have names.' ) #, required=True )
    parser.add_argument('-q','--encod_dim', type=int, default=None, help='Number of encoding dimensions, if None: runs hyperparameter optimization to determine best encoding dimension.')#, required=True)
    parser.add_argument('-sa','--sample_anno', default=None, type=str, help='Path to sample annotation file, must be comma separated, automatically selects sampleID column as the one which has input_file rownames in it.' )
    parser.add_argument('-cov','--covariates', default=None, nargs='*', help='List of covariate names in sample_anno, can either be columns filled with 0 and 1, multiple numbers or strings (will be automatically converted to one-hot encoded matrix for training).' )
    parser.add_argument('-p','--profile', default='outrider', choices=['outrider', 'protrider', 'pca'], help='Choose which pre-defined implementation should be used.')
    parser.add_argument('-o','--output', type=str, default=None, help='Folder to save results (xarray object, plots, result list), creates new folder if it does not exist, otherwise creates new folder within to save xarray object.')
    parser.add_argument('-fl','--float_type', default='float64', choices=['float32', 'float64'], help='Which float type should be used, highly advised to keep float64 which may take longer, but accuracy is strongly impaired by float32.')
    parser.add_argument('-cpu','--num_cpus', type=check_positive_int, default=1, help='Number of cpus used.')
    parser.add_argument('-m','--max_iter', type=check_positive_int, default=15, help='Number of maximal training iterations.')
    parser.add_argument('-v','--verbose', type=bool, default=False, help='Print additional output info during training.')
    parser.add_argument('-s','--seed', type=int, default=7, help='Seed used for training, negative values (e.g. -1) -> no seed set.')
    parser.add_argument('-op','--output_plots', type=bool, default=True, help='Outputs a collection of useful plots.')
    parser.add_argument('-ol','--output_list', type=bool, default=False, help='Outputs result in form of a long list.')
    parser.add_argument('--X_is_outlier', type=str, default=None, help='Path to a file (same shape as input) with values of 0|1 for injected outliers, automatically performs precision-recall on in.')

    parser.add_argument('-dis','--distribution', default=None, choices=['neg_bin', 'gaus'], help='[predefined in profile] distribution assumed for the measurement data.')
    parser.add_argument('-dt','--data_trans', default=None, choices=['sf', 'log', 'none'], help='[predefined in profile] change transformation scheme for measurement data during training.')
    parser.add_argument('-nf','--noise_factor', default=None, type=float, help='[predefined in profile] factor which defines the amount of noise applied for a denoising autoencoder model.')
    parser.add_argument('-ld','--loss_dis', default=None, choices=['neg_bin', 'gaus'], help='[predefined in profile] loss distribution used for training.')
    parser.add_argument('-pre','--prepro', default=None, choices=['none', 'sf_log'], help='[predefined in profile] preprocess data before input.')
    parser.add_argument('--batch_size', type=int, default=None, help='batch_size used for stochastic training. Default is to not train stochastically.')
    parser.add_argument('--parallelize_D', action='store_true', dest='parallelize_D', help='If this flag is given, parallelizes fitting of decoder per feature. Default behavior depends on sample size.')
    parser.add_argument('--no_parallelize_D', action='store_false', dest='parallelize_D', help='If this flag is given, decoder fit will not be parallelized per feature. Default behavior depends on sample size.')
    parser.set_defaults(parallelize_D=None)

    args = parser.parse_args(args_input)
    return vars(args)



def check_positive_int(value):
    ivalue = int(value)
    return check_positive(ivalue)

def check_positive(value):
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive value" )
    return value
    

class Check_parser():

    def __init__(self, args_input):

        ### modify arguments and transform to path obj
        self.args_mod = args_input.copy()
        self.args_mod['input'] = self.check_input_file(args_input['input'])
        self.args_mod['sample_anno'] = self.check_sample_anno(args_input['sample_anno'])
        self.args_mod['encod_dim'] = self.check_encod_dim(args_input['encod_dim'])
        self.args_mod['output'] = self.check_output(args_input['input'], args_input['output'])
        self.args_mod['X_is_outlier'] = self.check_X_is_outlier(args_input['X_is_outlier'])

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



    def check_X_is_outlier(self, X_is_outlier):
        if X_is_outlier is None:
            return None
        else:
            return self.get_path(X_is_outlier, "X_is_outlier")


