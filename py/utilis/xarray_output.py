import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import h5py



def xrds_to_zarr(xrds_obj, output_path):
    """
    separate method which perpares an xrds to be exported
    :param xrds_obj: xarray dataset object
    :param output_path: path output should be written in
    """
    xrds = xrds_obj.copy(deep=True)
    xrds.attrs["profile"] = [xrds.attrs["profile"].get_names()]

    ### None attributes can not be exported
    for at in xrds.attrs.keys():
        if xrds.attrs[at] is None:
            xrds.attrs[at] = "None"

    xrds.to_zarr(output_path, mode="w")


def xrds_to_list(xrds_obj, output_file, full_output=True):
    """
    transforms xarray object into a long list of samples_meas values
    :param xrds_obj: xarray dataset object
    :param output_file: filepath
    :param full_output: True if all values, False leads to truncated output with pvalue_adj < 0.5
    """
    sample_dir = {"sample_meas": [str(s) + "_" + str(g) for s in xrds_obj.coords["sample"].values for g in
                                  xrds_obj.coords["meas"].values],
                  "sample": np.repeat(xrds_obj.coords["sample"].values, len(xrds_obj.coords["meas"].values)),
                  "meas": np.tile(xrds_obj.coords["meas"].values, len(xrds_obj.coords["sample"].values)),
                  "logfc": xrds_obj["X_logfc"].values.flatten(),
                  "fc": np.exp(xrds_obj["X_logfc"].values.flatten()),
                  "pvalue": xrds_obj["X_pvalue"].values.flatten(),
                  "pvalue_adj": xrds_obj["X_pvalue_adj"].values.flatten(),
                  "z_score": xrds_obj["X_zscore"].values.flatten(),
                  "meas_raw": xrds_obj["X_raw"].values.flatten(),
                  "meas_trans": (xrds_obj["X_trans"] + xrds_obj["X_center_bias"]).values.flatten(),
                  "meas_trans_norm": ((xrds_obj["X_trans"] + xrds_obj["X_center_bias"]) - (
                          xrds_obj["X_trans_pred"] - xrds_obj["X_center_bias"])).values.flatten(),
                  }

    if "X_is_outlier" in xrds_obj:
        sample_dir["is_outlier"] = xrds_obj["X_is_outlier"].values.flatten()

    sample_dir["is_significant"] = xrds_obj["X_pvalue_adj"].values.flatten() < 0.05  # default considered as significant
    sample_outlier_list = pd.DataFrame(data=sample_dir)

    if full_output is False:
        sample_outlier_list = sample_outlier_list.loc[sample_outlier_list["pval_adj"] < 0.5,]
    sample_outlier_list.to_csv(output_file, index=False)


def xrds_to_tables(xrds_obj, output_path,
                   tables_to_dl=["X_trans", "X_trans_pred", "X_logfc", "X_pvalue", "X_pvalue_adj", "X_zscore"]):
    """
    outputs (multiple) .csv tables for selected matrices names
    :param xrds_obj: xrarry dataset object
    :param output_path: folder to write tables
    :param tables_to_dl:  list of tables to write
    """
    for ta in tables_to_dl:
        if ta == "X_trans":
            df = pd.DataFrame(data=(xrds_obj["X_trans"] + xrds_obj["X_center_bias"]).values, columns=xrds_obj.coords["meas"].values,
                              index=xrds_obj.coords["sample"].values)
        else:
            df = pd.DataFrame(data=xrds_obj[ta].values, columns=xrds_obj.coords["meas"].values,
                              index=xrds_obj.coords["sample"].values)
        df.to_csv(Path(output_path) / ta + '.csv', sep=",")




def xrds_to_hdf5(xrds_obj, output_path, hdf5_chunks=True, hdf5_compression="gzip", hdf5_compression_opts=9):
    """
    exports xarray object in hdf5 format, compressses it or saves in chunks,
    see https://docs.h5py.org/en/stable/
    :param xrds_obj: xarray object
    :param output_path: path output should be written in
    :param hdf5_chunks: chunk size or boolean (see hdf5 documentation)
    :param hdf5_compression: specify compression methods (see hdf5 documentation)
    :param hdf5_compression_opts: degree of compression (0-9) (see hdf5 documentation)
    """

    xrds = xrds_obj.copy(deep=True)

    with h5py.File(output_path, 'w') as hf:

        g1 = hf.create_group('X_tables')
        for i in [x_name for x_name in xrds.keys() if 'X' in x_name]:
            g1.create_dataset(i, data=xrds[i], chunks=hdf5_chunks, compression=hdf5_compression,
                              compression_opts=hdf5_compression_opts)

        g2 = hf.create_group('model_weights')
        for i in [x_name for x_name in xrds.keys() if 'coder' in x_name]:
            g2.create_dataset(i, data=xrds[i], chunks=hdf5_chunks, compression=hdf5_compression,
                              compression_opts=hdf5_compression_opts)

        g3 = hf.create_group('other_tables')
        for i in [x_name for x_name in xrds.keys() if 'coder' not in x_name and "X" not in x_name]:
            g3.create_dataset(i, data=np.string_(xrds[i]), chunks=hdf5_chunks, compression=hdf5_compression,
                              compression_opts=hdf5_compression_opts)

        g4 = hf.create_group('axis_labels')
        for i in xrds.coords:
            # axis labels sometimes too large to be squeezed in .attrs variables
            g4.create_dataset(i, data=np.string_(xrds.coords[i]), chunks=hdf5_chunks, compression=hdf5_compression,
                              compression_opts=hdf5_compression_opts)

        g5 = hf.create_group('metadata')
        for i in xrds.attrs:
            g5[i] = np.string_(str(xrds.attrs[i]))










