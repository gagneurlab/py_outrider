import argparse



def parse_args(args_input):
    parser = argparse.ArgumentParser(description='run outrider')
    parser.add_argument('--file_meas', type=str, help='path to measurement file, like gene counts, rows:samples x columns:genes') #, required=True ) TODO CHANGE
    parser.add_argument('--encod_dim', type=int, default=None, help='number of encoding dimensions, if None: runs hyperparameter optimization to determine best encoding dimension')#, required=True)
    parser.add_argument('--file_sa', default=None, type=str, help='sample annotation file' )
    parser.add_argument('--cov_used', default=None, nargs='*', help='list of covariate names in sample annotation file' )
    parser.add_argument('--profile', default='outrider', choices=['outrider', 'protrider', 'pca'], help='which implementation should be used')
    parser.add_argument('--output', type=str, default=None, help='file or folder to save results')
    parser.add_argument('--output_list', type=bool, default=False, help='output results in form of a long list')
    parser.add_argument('--float_type', default='float64', choices=['float32', 'float64'], help='which float type to be used, highly advised to keep float')
    parser.add_argument('--num_cpus', type=int, default=1, help='number of cpus used')
    parser.add_argument('--max_iter', type=int, default=15, help='number of maximal training iterations')
    parser.add_argument('--verbose', type=bool, default=False, help='print additional output info')
    parser.add_argument('--seed', type=int, default=None, help='seed used for training [NOT IMPLEMENTED]')
    parser.add_argument('--output_plots', type=bool, default=False, help='outputs a collection of useful plots')
    parser.add_argument('--X_is_outlier', type=str, default=None, help='path to 0|1 matrix of injected outlier, automatically performs precision-recall on in')

    parser.add_argument('--profile_distribution', default=None, choices=['neg_bin', 'gaus'], help='distribution assumed for the measurement data')
    parser.add_argument('--profile_norm', default=None, choices=['sf', 'log2', 'none'], help='change normalization scheme for measurement data')
    parser.add_argument('--profile_noise', default=None, type=float, help='noise applied for denoising autoencoder')

    args = parser.parse_args(args_input)
    return vars(args)




