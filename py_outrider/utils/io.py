import os
import numpy as np
import pandas as pd
import anndata

from .print_func import print_time


def write_output(adata, filename, filetype='h5ad'):
    assert filetype in ('h5ad', 'csv', 'zarr'), 'Unknown output file type'

    # if os.path.isdir(filename): # only check for existing dir
    #     filename = filename + "/py_outrider_results"

    if filetype == 'h5ad':
        ext = os.path.splitext(filename)[-1].lower()
        if ext != '.h5ad':
            filename = filename + '.h5ad'
        adata.write(filename)
    elif filetype == 'csv':
        adata.write_csvs(filename)
    elif filetype == 'zarr':
        adata.write_zarr(filename)


def write_results_table(adata, filename, all=True, alpha=0.05):

    print_time("Writing results table")
    sample_dir = {"sample": np.repeat(adata.obs_names, adata.n_vars),
                  "feature": np.tile(adata.var_names, adata.n_obs),
                  "pvalue": adata.layers["X_pvalue"].flatten(),
                  "padj": adata.layers["X_padj"].flatten(),
                  "raw_value": adata.layers["X_raw"].flatten(),
                  "preprocessed_value": adata.layers["X_prepro"].flatten(),
                  "predicted_value": adata.layers["X_predicted"].flatten(),
                  }

    effect_types = ["fc", "l2fc", "zscore", "delta"]
    for effect in effect_types:
        if "outrider_" + effect in adata.layers.keys():
            sample_dir[effect] = adata.layers["outrider_" + effect].flatten()

    sample_dir["aberrant"] = adata.layers["X_padj"].flatten() < alpha
    sample_outlier_list = pd.DataFrame(data=sample_dir)

    if all is False:
        sample_outlier_list = sample_outlier_list.loc[aberrant is True, ]
    sample_outlier_list.to_csv(filename, index=False)


def read_data(input_file, sample_anno_file=None, dtype='float64'):
    assert input_file is not None, 'input_file path specified is None'

    ext = os.path.splitext(input_file)[-1].lower()
    assert ext in ['.csv', '.h5ad'], (
        'input_file has to be either a csv or h5ad file')

    if ext == ".csv":
        input_data = pd.read_csv(input_file, sep=",", header=0,
                                 index_col=0).fillna(np.nan)
        adata = anndata.AnnData(X=input_data.values, dtype=dtype)
        adata.obs_names = input_data.index
        adata.var_names = input_data.columns
    else:
        adata = anndata.read(input_file)

    # use sample annotation if supplied
    if sample_anno_file is not None:
        adata = read_sample_anno(sample_anno_file, adata)
        # TODO consider known covariates

    return adata


def read_sample_anno(sample_anno_file, adata):
    assert sample_anno_file is not None, 'sample_anno_file specified is None'

    sample_anno = pd.read_csv(sample_anno_file, sep=",", header=0,
                              index_col=0).fillna(np.nan)
    adata.obs = sort_sample_anno(sample_anno, adata)

    return adata


def sort_sample_anno(sample_anno, adata):
    # find sample_id column
    sample_col_found = None
    for col in sample_anno:
        if set(adata.obs_names).issubset(sample_anno[col]):
            sample_col_found = col

    if sample_col_found is None:
        raise ValueError("input sample names not found in sample_anno"
                         " or not complete")
    elif len(sample_anno[sample_col_found]) != len(
                                        set(sample_anno[sample_col_found])):
        raise ValueError((
                          "duplicates found in sample_anno ",
                          f"sample_id column: {sample_col_found}"))
    else:
        sample_anno.rename(columns={sample_col_found: "sample_id"},
                           inplace=True)
        sample_anno.set_index(sample_anno["sample_id"], inplace=True)

    # sort according to X_file and remove unnecessary
    sample_anno = sample_anno.reindex(adata.obs_names)

    return sample_anno


def create_adata_from_arrays(X_input, sample_anno=None, dtype='float64'):

    adata = anndata.AnnData(X=X_input, dtype=dtype)

    # use sample annotation if supplied
    if sample_anno is not None:
        adata.obs = sort_sample_anno(sample_anno, adata)

    return adata
