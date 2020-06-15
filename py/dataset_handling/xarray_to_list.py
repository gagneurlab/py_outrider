# import pandas as pd
# import numpy as np
# import xarray as xr
#
# from utilis_methods import data_handling as dh
#
#
# ### useful help methods to deal with xarray objects - outputs useable tables
# ### not directly used in omics-OUTRIDER
#
#
# def xrds_to_list(file_path, output_file, full_output=False):
#     xrds = xr.open_zarr(file_path)
#     sample_dir = {"sample_gene": [str(s) + "_" + str(g) for s in xrds.coords["samples"].values for g in
#                                   xrds.coords["genes"].values],
#                   "log2fc": xrds["log2fc"].values.flatten(),
#                   "pval": xrds["pval"].values.flatten(),
#                   "pval_adj": xrds["pval_adj"].values.flatten(),
#                   "z_score": xrds["z_score"].values.flatten()
#                   }
#     sample_outlier_list = pd.DataFrame(data=sample_dir)
#
#     if full_output is False:
#         sample_outlier_list = sample_outlier_list.loc[sample_outlier_list["pval_adj"] < 0.5,]
#         sample_outlier_list.reset_index(inplace=True, drop=True)
#     sample_outlier_list.to_csv(output_file, index=False)
#
#
# 
#
# def xrds_to_tables(file_path, output_path):
#     xrds = xr.open_zarr(file_path)
#     tables_to_dl = ["counts","counts_pred","log2fc","pval","pval_adj","z_score"]
#     for ta in tables_to_dl:
#         df = pd.DataFrame(data=xrds[ta].values, columns = xrds["genes"].values, index=xrds["samples"].values )
#         df.to_csv(dh.path(output_path, ta+'.csv'), delimiter=",")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
