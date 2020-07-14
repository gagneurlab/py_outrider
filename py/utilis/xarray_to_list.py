import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path


def xrds_to_list(xrds, output_file, full_output=True):
    """
    transforms xarray object into a long list of samples_meas values
    :param xrds: xarray dataset object
    :param output_file: filepath
    :param full_output: True if all values, False leads to truncated output with pvalue_adj < 0.5
    """
    sample_dir = {"sample_meas": [str(s) + "_" + str(g) for s in xrds.coords["sample"].values for g in
                                  xrds.coords["meas"].values],
                  "sample": np.repeat(xrds.coords["sample"].values, len(xrds.coords["meas"].values)),
                  "meas": np.tile(xrds.coords["meas"].values, len(xrds.coords["sample"].values)),
                  "log2fc": xrds["X_log2fc"].values.flatten(),
                  "fc": np.power(2, xrds["X_log2fc"].values.flatten()),
                  "pvalue": xrds["X_pvalue"].values.flatten(),
                  "pvalue_adj": xrds["X_pvalue_adj"].values.flatten(),
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


def xrds_to_tables(xrds, output_path,
                   tables_to_dl=["X_trans", "X_trans_pred", "X_log2fc", "X_pvalue", "X_pvalue_adj", "X_zscore"]):
    """
    outputs .csv tables for selected matrices names
    :param xrds: xrarry dataset object
    :param output_path: folder to write tables
    :param tables_to_dl:  list of tables to write
    """
    for ta in tables_to_dl:
        if ta == "X_trans":
            df = pd.DataFrame(data=(xrds["X_trans"] + xrds["X_center_bias"]).values, columns=xrds.coords["meas"].values,
                              index=xrds.coords["sample"].values)
        else:
            df = pd.DataFrame(data=xrds[ta].values, columns=xrds.coords["meas"].values,
                              index=xrds.coords["sample"].values)
        df.to_csv(Path(output_path) / ta + '.csv', sep=",")
