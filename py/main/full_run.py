
from parser import check_parser
from dataset_handling.create_xarray import Create_xarray
import utilis.stats_func as st
from dataset_handling.model_dataset import Model_dataset
from dataset_handling.custom_output import Custom_output
from hyperpar_opt import Hyperpar_opt





class Full_run():

    def __init__(self, args_input):
        args_mod = check_parser.Check_parser(args_input).args_mod
        print('parser check')
        xrds = Create_xarray(args_mod).xrds
        print('xarray created')
        print(xrds)


        model_ds = Model_dataset(xrds)
        print('dataset created')
        # model_ds.inject_outlier(inj_freq=1e-3, inj_mean=3, inj_sd=1.6)
        # print('outlier injected')


        if xrds.attrs["encod_dim"] is None:
            print('encod_dim is None -> running hyperpar-opt')
            Hyperpar_opt(model_ds)

        print('noise injected')
        model_ds.inject_noise(inj_freq=1, inj_mean=0, inj_sd=1)


        fit_model = model_ds.profile.fit_model(model_ds)
        print('run model')
        xrds = fit_model.run_model_fit()
        print(xrds)

        if "X_is_outlier" in xrds:
            pre_rec = st.get_prec_recall(xrds["X_pvalue"].values, xrds["X_is_outlier"].values)
            print(f'precision-recall: { pre_rec["auc"] }')


        ### export
        print(xrds)
        xrds.attrs["profile"] = xrds.attrs["profile"].__class__.__name__
        xrds.to_zarr(xrds.attrs["output"] , mode="w")

        print('start plotting')
        Custom_output(xrds)





        print('finished')












