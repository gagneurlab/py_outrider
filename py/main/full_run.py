
from parser import check_parser
from dataset_handling.create_xarray import Create_xarray
import utilis.stats_func as st
from dataset_handling.model_dataset import Model_dataset
from dataset_handling.custom_output import Custom_output
from utilis.xarray_output import xrds_to_zarr
from hyperpar_opt import Hyperpar_opt
from utilis.print_func import print_time




class Full_run():

    def __init__(self, args_input):
        print_time('parser check for correct input arguments')
        args_mod = check_parser.Check_parser(args_input).args_mod

        print_time('create xarray object out of input data')
        xrds = Create_xarray(args_mod).xrds
        print(xrds)


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

        print_time('save model dataset to file')
        xrds_to_zarr(xrds_obj=xrds, output_path=xrds.attrs["output"])


        if "X_is_outlier" in xrds:
            pre_rec = st.get_prec_recall(xrds["X_pvalue"].values, xrds["X_is_outlier"].values)
            print_time(f'precision-recall: { pre_rec["auc"] }')


        print_time('start creating custom output if specified')
        Custom_output(xrds)


        print_time('finished whole run')










