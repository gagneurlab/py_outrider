# py_outrider

py_outrider is a flexible framework for outlier detection in omics datasets.
it is currently implemented as a collection of modules which can be plugged together, 
but in any case it should be converted into a package,
while still being able to run it on the command line as follows:


# How to start

After cloning this repository, you can set up your own environment with all required packages using the `environment.yml`
or you run it using the following conda environment: `conda activate outrider2`

(It is recommended to use `Tensorflow`>2.0.2, which fixed some rather significant runtime issues.)

Then, run `pip install -e /path/to/py_outrider_root_dir/` to install py_outrider in this environment. 
To check available options, run
```sh
py_outrider --help
```

# OUTRIDER
For large data sets, the python version has a way faster fitting procedure of outrider as the R version
```sh
py_outrider --input count_table.csv --encod_dim 5 --profile outrider --output OUTPUT_DIR --num_cpus 10 
```

> Be careful: input matrix must have the format: rows:samples x columns:genes, must be comma separated, needs 1. row and 1. column to have names


# PROTRIDER
More accurate (but with longer runtime) fitting method for proteins with covariate consideration
```sh
py_outrider --input protein_intensities.csv --encod_dim 5 --profile protrider --output OUTPUT_DIR --num_cpus 10 --sample_anno sample_anno.csv -cov batch gender --output_list True  --output_plots True
```

> Missing values due to missing detection in the mass-spectrometry must be declared as such, e.g. NAN and must not be kept 0, as this will be considered as true value.

(xarray dataset object can be transformed to `.hdf` and read in as PROTRIDER object for plotting, see [script](https://gitlab.cmm.in.tum.de/yepez/proteome_analysis/-/blob/master/Scripts/MultiOmics/Aberrant_Expression/PROTRIDER/hdf5_to_se.R) )



---
The full parser and possibilites are:

## Usage
```
usage: py_outrider [-h] [-in INPUT] [-q ENCOD_DIM] [-sa SAMPLE_ANNO]
                   [-cov [COVARIATES [COVARIATES ...]]]
                   [-p {outrider,protrider,pca}] [-o OUTPUT]
                   [-fl {float32,float64}] [-cpu NUM_CPUS] [-m MAX_ITER]
                   [-v VERBOSE] [-s SEED] [-op OUTPUT_PLOTS] [-ol OUTPUT_LIST]
                   [--X_is_outlier X_IS_OUTLIER] [-dis {neg_bin,gaus}]
                   [-dt {sf,log,none}] [-nf NOISE_FACTOR] [-ld {neg_bin,gaus}]
                   [-pre {none,sf_log}] [--batch_size BATCH_SIZE]
                   [--parallelize_D] [--no_parallelize_D]
```
## Arguments
### Quick reference table
|Short |Long                |Default   |Description                                                                                                                                                   |
|------|----------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`-h`  |`--help`            |          |Show this help message and exit.                                                                                                                               |
|`-in` |`--input`           |`None`    |Path to a file containing the input data matrix, like a gene count table, format: rows:samples x columns:genes, must be comma separated, needs 1. row and 1. column to have names.  |
|`-q`  |`--encod_dim`       |`None`    |Number of encoding dimensions, if None: runs hyperparameter optimization to determine best encoding dimension.                                                 |
|`-sa` |`--sample_anno`     |`None`    |Path to sample annotation file, must be comma separated, automatically selects sampleID column as the one which has input_file rownames in it.                                       |
|`-cov`|`--covariates`      |`None`    |List of covariate names in sample_anno, can either be columns filled with 0 and 1, multiple numbers or strings (will be automatically converted to one-hot encoded matrix for training). |                                                                             |
|`-p`  |`--profile`         |`outrider`|Choose which pre-defined implementation should be used.                                                                                                        |
|`-o`  |`--output`          |`None`    |Folder to save results (xarray object, plots, result list), creates new folder if it does not exist, otherwise creates new folder within to save xarray object.|
|`-fl` |`--float_type`      |`float64` |Which float type should be used, highly advised to keep float64 which may take longer, but accuracy is strongly impaired by float32.                           |
|`-cpu`|`--num_cpus`        |`1`       |Number of cpus used.                                                                                                                                           |
|`-m`  |`--max_iter`        |`15`      |Number of maximal training iterations.                                                                                                                         |
|`-v`  |`--verbose`         |          |Print additional output info during training.                                                                                                                  |
|`-s`  |`--seed`            |`7`       |Seed used for training, negative values (e.g. -1) -> no seed set.                                                                                              |
|`-op` |`--output_plots`    |          |Outputs a collection of useful plots.                                                                                                                          |
|`-ol` |`--output_list`     |          |Outputs result in form of a long list.                                                                                                                         |
|      |`--X_is_outlier`    |`None`    |Path to a measurement file with values of 0|1 for injected outliers, automatically performs precision-recall on in.                                            |
|`-dis`|`--distribution`    |`None`    |Distribution assumed for the measurement data [predefined in profile].                                                                                         |
|`-dt` |`--data_trans`      |`None`    |Change transformation scheme for measurement data during training [predefined in profile].                                                                     |
|`-nf` |`--noise_factor`    |`None`    |Factor which defines the amount of noise applied for a denoising autoencoder model [predefined in profile].                                                    |
|`-ld` |`--loss_dis`        |`None`    |Loss distribution used for training [predefined in profile].                                                                                                   |
|`-pre`|`--prepro`          |`None`    |Preprocess data before input [predefined in profile].                                                                                                          |
|      |`--batch_size`      |`None`    |batch_size used for stochastic training. Default is to not train stochastically.                                                                               |
|      |`--parallelize_D`   |          |If this flag is given, parallelizes fitting of decoder per feature. Default behavior depends on sample size.                                                  |
|      |`--no_parallelize_D`|          |If this flag is given, decoder fit will not be parallelized per feature. Default behavior depends on sample size.                                             |


