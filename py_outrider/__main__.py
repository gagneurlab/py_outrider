import sys, os 
# dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(dir_path) # print(dir_path)
from pathlib import Path

from py_outrider.parser import input_parser
from py_outrider.utils.print_func import print_dict
from py_outrider.parser import check_parser
from py_outrider.dataset_handling.create_xarray import Create_xarray
import py_outrider.utils.stats_func as st
from py_outrider.dataset_handling.model_dataset import Model_dataset
# from py_outrider.dataset_handling.custom_output import Custom_output # if uncommented reticulate wont recognize this file as a module for some reason
from py_outrider.utils.xarray_output import xrds_to_zarr
from py_outrider.hyperpar_opt import Hyperpar_opt
from py_outrider.utils.print_func import print_time


def main():
    args = input_parser.parse_args(sys.argv[1:])

    # print(args)
    if args['file_meas'] is not None:    #
        full_run(args_input=args)

    else:
        ### test run options for protrider | outrider on sample data
        test_dir = Path(__file__).parent.absolute().parents[0] / "tests"

        # OUTRIDER test run (RNA-seq gene counts):
        sample_args = { "file_meas" : test_dir / "sample_gene.csv", "encod_dim": 10,
                  'verbose':False, 'num_cpus':5, 'seed':5, "output_plots":True, "output_list":True,
                  "max_iter": 3, "profile": "outrider",
                  # 'file_sa': test_dir / 'sample_gene_sa.csv', 'covariates': ["is_male", "batch"],
                 "output": test_dir / "sample_gene_output"
                  }
                  
        # PROTRIDER test run (protein MS intensities):
        # sample_args = { "file_meas" : test_dir / "sample_protein.csv", "encod_dim": 10,
        #           'verbose':True, 'num_cpus':5, 'seed':5, "output_plots":True, "output_list":True,
        #           "max_iter": 3, "profile": "protrider",
        #           'file_sa': test_dir / 'sample_protein_sa.csv', 'covariates': ["is_male", "batch"],
        #          "output": test_dir / "sample_protein_output"
        #           }

        args.update(sample_args)
        # print_dict(args)
        full_run(args_input=args)


def full_run(args_input):
    print_time('parser check for correct input arguments')
    args_mod = check_parser.Check_parser(args_input).args_mod

    print_time('create xarray object out of input data')
    xrds = Create_xarray(args_mod).xrds
    print(xrds)

    # run outrider model
    xrds = run_outrider(xrds)

    # output
    print_time('save model dataset to file')
    xrds_to_zarr(xrds_obj=xrds, output_path=xrds.attrs["output"])

    if "X_is_outlier" in xrds:
        pre_rec = st.get_prec_recall(xrds["X_pvalue"].values, xrds["X_is_outlier"].values)
        print_time(f'precision-recall: { pre_rec["auc"] }')

    # print_time('start creating custom output if specified')
    # Custom_output(xrds)

    print_time('finished whole run')
    
def run_from_R_OUTRIDER(X_input, sample_anno_input, args_input):
    print_time('parser check for correct input arguments')
    args_input = input_parser.parse_args(args_input)

    print_time('create xarray object out of input data')
    xrds = Create_xarray(X_input=X_input, sample_anno_input=sample_anno_input, args_input=args_input).xrds
    print(xrds)
    print(xrds.attrs["profile"].get_names())
    # print(f'pval distribution: { xrds.attrs["profile"].dis }')
    # print(f'loss distribution: { xrds.attrs["profile"].loss_dis }')
    # print(f'preprocessing: { xrds.attrs["profile"].prepro }')
    # print(f'transformation: { xrds.attrs["profile"].data_trans }')

    # run outrider model
    xrds = run_outrider(xrds)

    print_time('finished whole run')
    return xrds


def run_outrider(xrds):
    print_time('create model dataset class')
    model_ds = Model_dataset(xrds)

    if xrds.attrs["encod_dim"] is None:
        print_time('encod_dim is None -> run hyperparameter optimisation')
        Hyperpar_opt(model_ds)

    print_time('inject noise if specified')
    model_ds.inject_noise(inj_freq=1, inj_mean=0, inj_sd=1)

    fit_model = model_ds.profile.fit_model(model_ds)
    print_time('start running model fit')
    xrds = fit_model.run_model_fit()

    return xrds









