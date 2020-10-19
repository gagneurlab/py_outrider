import argparse


def parse_args(args_input):
    parser = argparse.ArgumentParser(description='run outrider')
    parser.add_argument('-in','--file_meas', type=str, help='path to a measurement file, like a gene count table, format: rows:samples x columns:genes, must be comma separated, needs 1. row and 1. column to have names' ) #, required=True )
    parser.add_argument('-q','--encod_dim', type=int, default=None, help='number of encoding dimensions, if None: runs hyperparameter optimization to determine best encoding dimension')#, required=True)
    parser.add_argument('-sa','--file_sa', default=None, type=str, help='path to sample annotation file, must be comma separated, automatically selects colum which has file_meas rownames in it' )
    parser.add_argument('-cov','--covariates', default=None, nargs='*', help='list of covariate names in file_sa, can either be columns filled with 0 and 1, multiple numbers or strings (will be automatically converted to one-hot encoded matrix for training)' )
    parser.add_argument('-p','--profile', default='outrider', choices=['outrider', 'protrider', 'pca'], help='choose which pre-defined implementation should be used')
    parser.add_argument('-o','--output', type=str, default=None, help='folder to save results (xarray object, plots, result list), creates new folder if it does not exist, otherwise creates new folder within to save xarray object')
    parser.add_argument('-fl','--float_type', default='float64', choices=['float32', 'float64'], help='which float type should be used, highly advised to keep float64 which may take longer, but accuracy is strongly impaired by float32')
    parser.add_argument('-cpu','--num_cpus', type=check_positive_int, default=1, help='number of cpus used')
    parser.add_argument('-m','--max_iter', type=check_positive_int, default=15, help='number of maximal training iterations')
    parser.add_argument('-v','--verbose', type=bool, default=False, help='print additional output info during training')
    parser.add_argument('-s','--seed', type=int, default=7, help='seed used for training, negative values (e.g. -1) -> no seed set')
    parser.add_argument('-op','--output_plots', type=bool, default=True, help='outputs a collection of useful plots')
    parser.add_argument('-ol','--output_list', type=bool, default=False, help='outputs result in form of a long list')
    parser.add_argument('--X_is_outlier', type=str, default=None, help='path to a measurement file with values of 0|1 for injected outliers, automatically performs precision-recall on in')

    parser.add_argument('-dis','--distribution', default=None, choices=['neg_bin', 'gaus'], help='[predefined in profile] distribution assumed for the measurement data')
    parser.add_argument('-dt','--data_trans', default=None, choices=['sf', 'log', 'none'], help='[predefined in profile] change transformation scheme for measurement data during training')
    parser.add_argument('-nf','--noise_factor', default=None, type=float, help='[predefined in profile] factor which defines the amount of noise applied for a denoising autoencoder model')
    parser.add_argument('-ld','--loss_dis', default=None, choices=['neg_bin', 'gaus'], help='[predefined in profile] loss distribution used for training')
    parser.add_argument('-pre','--prepro', default=None, choices=['none', 'sf_log'], help='[predefined in profile] preprocess data before input')

    args = parser.parse_args(args_input)
    return vars(args)



def check_positive_int(value):
    ivalue = int(value)
    return check_positive(ivalue)

def check_positive(value):
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive value" )
    return value


