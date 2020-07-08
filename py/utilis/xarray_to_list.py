import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path


def xrds_to_list(xrds, output_file, full_output=True):
    sample_dir = {"sample_meas": [str(s) + "_" + str(g) for s in xrds.coords["sample"].values for g in
                                  xrds.coords["meas"].values],
                  "sample": np.repeat(xrds.coords["sample"].values, len(xrds.coords["meas"].values)),
                  "meas": np.tile(xrds.coords["meas"].values, len(xrds.coords["sample"].values)),
                  "log2fc": xrds["X_log2fc"].values.flatten(),
                  "fc": np.power(2, xrds["X_log2fc"].values.flatten()),
                  "pval": xrds["X_pvalue"].values.flatten(),
                  "pval_adj": xrds["X_pvalue_adj"].values.flatten(),
                  "z_score": xrds["X_zscore"].values.flatten(),
                  "meas_raw": xrds["X_raw"].values.flatten(),
                  "meas_trans": (xrds["X_trans"] + xrds["X_center_bias"]).values.flatten(),
                  "meas_trans_norm": ((xrds["X_trans"] + xrds["X_center_bias"]) - (
                              xrds["X_trans_pred"] - xrds["X_center_bias"])).values.flatten(),
                  }

    if "X_is_outlier" in xrds:
        sample_dir["is_outlier"] = xrds["X_is_outlier"].values.flatten()

    sample_outlier_list = pd.DataFrame(data=sample_dir)

    if full_output is False:
        sample_outlier_list = sample_outlier_list.loc[sample_outlier_list["pval_adj"] < 0.5,]
    sample_outlier_list.to_csv(output_file, index=False)





def xrds_to_tables(xrds, output_path):
    ### seperate case
    df = pd.DataFrame(data=(xrds["X_trans"]+xrds["X_center_bias"]).values, columns = xrds.coords["meas"].values, index=xrds.coords["sample"].values )
    df.to_csv(Path(output_path) / 'X_trans.csv', delimiter=",")

    tables_to_dl = ["X_trans_pred","X_log2fc","X_pvalue","X_pvalue_adj","X_zscore"]
    for ta in tables_to_dl:
        df = pd.DataFrame(data=xrds[ta].values, columns = xrds.coords["meas"].values, index=xrds.coords["sample"].values )
        df.to_csv(Path(output_path) / ta+'.csv', delimiter=",")





























