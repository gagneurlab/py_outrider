import sys, os 
from pathlib import Path

from . import parser
from .outrider import outrider
from .hyperpar_opt import Hyperpar_opt
from .utils.print_func import print_time #, print_info_run
from .utils.io import read_data, create_adata_from_arrays, write_output, write_results_table

def main():
    args = parser.parse_args(sys.argv[1:])

    if args['input'] is not None:    #
        full_run(args_input=args)

    else:
        ### test run options for protrider | outrider on sample data
        test_dir = Path(__file__).parent.absolute().parents[0] / "tests"

        # OUTRIDER test run (RNA-seq gene counts):
        sample_args = { "input" : test_dir / "sample_gene.csv", "encod_dim": 10,
                  'verbose':False, 'num_cpus':5, 'seed':5, "output_plots":True, "output_list":True,
                  "max_iter": 3, "profile": "outrider",
                  # 'sample_anno': test_dir / 'sample_gene_sa.csv', 'covariates': ["is_male", "batch"],
                 "output": test_dir / "sample_gene_output"
                  }
                  
        # PROTRIDER test run (protein MS intensities):
        # sample_args = { "input" : test_dir / "sample_protein.csv", "encod_dim": 10,
        #           'verbose':True, 'num_cpus':5, 'seed':5, "output_plots":True, "output_list":True,
        #           "max_iter": 3, "profile": "protrider",
        #           'sample_anno': test_dir / 'sample_protein_sa.csv', 'covariates': ["is_male", "batch"],
        #          "output": test_dir / "sample_protein_output"
        #           }

        args.update(sample_args)
        full_run(args_input=args)


def full_run(args_input):
    print_time('parser check for correct input arguments')
    args = parser.Check_parser(args_input).args_mod

    print_time('create adata object out of input data')
    adata = read_data(args['input'], args['sample_anno'], args['float_type'])
    outrider_args = parser.extract_outrider_args(args) 
    print(outrider_args)
    # TODO: print_run_info(adata, outrider_args, args['profile'])
    
    # check need for hyper param opt
    if outrider_args["encod_dim"] is None:
        print_time('encod_dim is None -> run hyperparameter optimisation')
        hyper = Hyperpar_opt(adata, **outrider_args)
        outrider_args["encod_dim"] = hyper.best_encod_dim
        outrider_args["noise_factor"] = hyper.best_noise_factor

    # run outrider model
    adata = outrider(adata, **outrider_args)

    # output
    print_time('Saving result AnnData object to file')
    write_output(adata, filetype=args["output_type"], filename=args["output"])
    
    if args["output_res_table"] is not None:
        write_results_table(adata, filename=args["output_res_table"])

    print_time('finished whole run')

    
def run_from_R_OUTRIDER(X_input, sample_anno_input, args_input):
    print_time('parser check for correct input arguments')
    args = parser.parse_args(args_input)

    print_time('create adata object out of input data')
    adata = create_adata_from_arrays(X_input, sample_anno = sample_anno_input, dtype=args['float_type'])
    outrider_args = parser.extract_outrider_args(args)
    # TODO: print_run_info(adata, outrider_args, args['profile'])
    
    # check need for hyper param opt
    if outrider_args["encod_dim"] is None:
        print_time('encod_dim is None -> run hyperparameter optimisation')
        hyper = Hyperpar_opt(adata, outrider_args)
        outrider_args["encod_dim"] = hyper.best_encod_dim
        outrider_args["noise_factor"] = hyper.best_noise_factor

    # run outrider model
    adata = outrider(adata, outrider_args)

    print_time('finished whole run')
    return adata

