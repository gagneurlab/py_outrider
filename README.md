# py_outrider

py_outrider is a flexible framework for outlier detection in omics datasets.
it is currently implemented as a collection of modules which can be plugged together, 
but in any case it should be converted into a package,
while still being able to run it on the command line as follows:

---
# how to start

before running, you might have to set your python path to the right directory:
```sh
export PYTHONPATH="/home/user/gitlab/py_outrider/py"
```
you can set up your own environment with all required packages in `requirements.txt`
or you run it using the following conda environment: `conda activate loipfi_tf2`

(it is recommended to use `Tensorflow`>2.0.2, which fixed some rather significant runtime issues)

---
# OUTRIDER
for large data sets, the python version has a way faster fitting procedure of outrider as the R version
```sh
python /home/user/gitlab/py_outrider/py/main/main.py --file_meas count_table.csv --encod_dim 5 --profile outrider --output OUTPUT_DIR --num_cpus 10 
```

> be careful: input matrix file_meas must have the format: rows:samples x columns:genes, must be comma separated, needs 1. row and 1. column to have names

---
# PROTRIDER
more accurate (but with longer runtime) fitting method for proteins with covariate consideration
```sh
python /home/user/gitlab/py_outrider/py/main/main.py --file_meas protein_intensities.csv --encod_dim 5 --profile protrider --output OUTPUT_DIR --num_cpus 10 --file_sa sample_anno.csv -cov batch gender --output_list True  --output_plots True
```

> missing values due to missing detection in the mass-spectrometry must be declared as such, e.g. NAN and must not be kept 0, as this will be considered as true value.

(xarray dataset object can be transformed to `.hdf` and read in as PROTRIDER object for plotting, see [script](https://gitlab.cmm.in.tum.de/yepez/proteome_analysis/-/blob/master/Scripts/MultiOmics/Aberrant_Expression/PROTRIDER/hdf5_to_se.R) )



---
the full parser and possibilites are:

## Usage
```
usage: py_outrider/py/main/main.py [-h] [-in FILE_MEAS] [-q ENCOD_DIM] [-sa FILE_SA]
               [-cov [COVARIATES [COVARIATES ...]]]
               [-p {outrider,protrider,pca}] [-o OUTPUT]
               [-fl {float32,float64}] [-cpu NUM_CPUS] [-m MAX_ITER]
               [-v VERBOSE] [-s SEED] [-op OUTPUT_PLOTS] [-ol OUTPUT_LIST]
               [--X_is_outlier X_IS_OUTLIER] [-dis {neg_bin,gaus}]
               [-dt {sf,log,none}] [-nf NOISE_FACTOR] [-ld {neg_bin,gaus}]
               [-pre {none,sf_log}]
```
## Arguments
### Quick reference table
|Short |Long            |Default   |Description                                                                                                                                                   |
|------|----------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`-h`  |`--help`        |          |show this help message and exit                                                                                                                               |
|`-in` |`--file_meas`   |`None`    |path to a measurement file, like a gene count table, format: rows:samples x columns:genes, must be comma separated, needs 1. row and 1. column to have names  |
|`-q`  |`--encod_dim`   |`None`    |number of encoding dimensions, if None: runs hyperparameter optimization to determine best encoding dimension                                                 |
|`-sa` |`--file_sa`     |`None`    |path to sample annotation file, must be comma separated, automatically selects colum which has file_meas rownames in it                                       |
|`-cov`|`--covariates`  |`None`    |list of covariate names in file_sa, can either be columns filled with 0 and 1, multiple numbers or strings (will be automatically converted to one-hot encoded matrix for training) |                                                                             |
|`-p`  |`--profile`     |`outrider`|choose which pre-defined implementation should be used                                                                                                        |
|`-o`  |`--output`      |`None`    |folder to save results (xarray object, plots, result list), creates new folder if it does not exist, otherwise creates new folder within to save xarray object|
|`-fl` |`--float_type`  |`float64` |which float type should be used, highly advised to keep float64 which may take longer, but accuracy is strongly impaired by float32                           |
|`-cpu`|`--num_cpus`    |`1`       |number of cpus used                                                                                                                                           |
|`-m`  |`--max_iter`    |`15`      |number of maximal training iterations                                                                                                                         |
|`-v`  |`--verbose`     |          |print additional output info during training                                                                                                                  |
|`-s`  |`--seed`        |`7`       |seed used for training, negative values (e.g. -1) -> no seed set                                                                                              |
|`-op` |`--output_plots`|          |outputs a collection of useful plots                                                                                                                          |
|`-ol` |`--output_list` |          |outputs result in form of a long list                                                                                                                         |
|      |`--X_is_outlier`|`None`    |path to a measurement file with values of 0|1 for injected outliers, automatically performs precision-recall on in                                            |
|`-dis`|`--distribution`|`None`    |distribution assumed for the measurement data [predefined in profile]                                                                                         |
|`-dt` |`--data_trans`  |`None`    |change transformation scheme for measurement data during training [predefined in profile]                                                                     |
|`-nf` |`--noise_factor`|`None`    |factor which defines the amount of noise applied for a denoising autoencoder model [predefined in profile]                                                    |
|`-ld` |`--loss_dis`    |`None`    |loss distribution used for training [predefined in profile]                                                                                                   |
|`-pre`|`--prepro`      |`None`    |preprocess data before input [predefined in profile]                                                                                                          |


