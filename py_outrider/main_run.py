from . import parser
from .hyperpar_opt import Hyperpar_opt
from .outrider import outrider
from .utils import print_func
from .utils.io import (read_data, create_adata_from_arrays, write_output,
                       write_results_table)


def full_run(args_input):
    print_func.print_time('parser check for correct input arguments')
    args = parser.Check_parser(args_input).args_mod

    print_func.print_time('parsed the following settings:')
    outrider_args = parser.extract_outrider_args(args)
    print(outrider_args)

    print_func.print_time('create adata object out of input data')
    adata = read_data(args['input'], args['sample_anno'],
                      outrider_args['float_type'])

    # check need for hyper param opt
    if outrider_args["encod_dim"] is None:
        print_func.print_time('encod_dim is None -> '
                              'run hyperparameter optimisation')
        hyper_args = outrider_args.copy()
        # set iterations during hyper param opt
        hyper_args["iterations"] = args["max_iter_hyper"]
        # set convergence limit during hyper param opt
        hyper_args["convergence"] = args["convergence_hyper"]
        hyper = Hyperpar_opt(adata, **hyper_args)
        adata.uns["hyperpar_table"] = hyper.hyperpar_table
        outrider_args["encod_dim"] = hyper.best_encod_dim
        outrider_args["noise_factor"] = hyper.best_noise_factor

    # run outrider model
    adata = outrider(adata, do_call_outliers=True, **outrider_args)

    # output
    print_func.print_time(
        f'Saving result AnnData object to file {args["output"]}')
    write_output(adata, filetype=args["output_type"], filename=args["output"])

    if args["output_res_table"] is not None:
        write_results_table(adata, filename=args["output_res_table"])

    print_func.print_time('finished whole run')


def run_from_R_OUTRIDER(X_input, sample_anno_input, args_input):
    print_func.print_time('parser check for correct input arguments')
    args = parser.parse_args(args_input)

    print_func.print_time('parsed the following settings:')
    outrider_args = parser.extract_outrider_args(args)
    print(outrider_args)

    print_func.print_time('create adata object out of input data')
    adata = create_adata_from_arrays(X_input, sample_anno=sample_anno_input,
                                     dtype=outrider_args['float_type'])

    # check need for hyper param opt
    if outrider_args["encod_dim"] is None:
        print_func.print_time(
            'encod_dim is None -> run hyperparameter optimisation')
        hyper_args = outrider_args.copy()
        # set iterations during hyper param opt
        hyper_args["iterations"] = args["max_iter_hyper"]
        # set convergence limit during hyper param opt
        hyper_args["convergence"] = args["convergence_hyper"]
        hyper = Hyperpar_opt(adata, **hyper_args)
        adata.uns["hyperpar_table"] = hyper.hyperpar_table
        outrider_args["encod_dim"] = hyper.best_encod_dim
        outrider_args["noise_factor"] = hyper.best_noise_factor

    # run outrider model
    adata = outrider(adata, do_call_outliers=False, **outrider_args)

    print_func.print_time('finished whole run')
    return adata
