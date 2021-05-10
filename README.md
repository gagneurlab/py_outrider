# py_outrider

py_outrider is a flexible framework for outlier detection in omics datasets.

# How to start

After cloning this repository, you can set up your own environment with all required packages using the `environment.yml`
or you run it using the following conda environment: `conda activate outrider2`

(It is recommended to use `Tensorflow`>2.0.2, which fixed some rather significant runtime issues.)

Then, run `pip install py_outrider` to install py_outrider in this environment (or `pip install git+https://github.com/gagneurlab/py_outrider.git` to install from github). 
To check available options, run
```sh
py_outrider --help
```

# OUTRIDER
For large data sets, the python version has a way faster fitting procedure of outrider as the R version
```sh
py_outrider --input count_table.csv --encod_dim 5 --profile outrider --output OUTPUT_FILE --num_cpus 10 
```

> Be careful: input matrix must have the format: rows:samples x columns:genes, must be comma separated (csv), and needs the 1st row and 1st column to contain the feature and sample names, respectively


# PROTRIDER
More accurate (but with longer runtime) fitting method for proteins with covariate consideration
```sh
py_outrider --input protein_intensities.csv --encod_dim 5 --profile protrider --output OUTPUT_FILE --num_cpus 10 --sample_anno sample_anno.csv -cov batch gender --output_res_table RESULTS_TABLE_FILE
```

> Missing values due to missing detection in the mass-spectrometry must be declared as such, e.g. NAN and must not be kept 0, as this will be considered as true value.

# Output
py_outrider outputs an AnnData object containing the data, fit results, p values etc. (by default as an .h5ad file). 
To read in the output file, and access e.g. adjusted pvalues, run
```python
import anndata
adata = anndata.read("/path/to/py_outrider_results.h5ad")
adata.layers["X_padj"]
```

Output matrices include:

|Name                  |Description                                                                                                                                     |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
|`X_pvalue`            |calculated p values                                                                                                                             |
|`X_padj`              |caluclated adjusted p values                                                                                                                    |
|`outrider_fc`         |calculated fold-changes                                                                                                                         |
|`outrider_l2fc`       |calculated log2 fold-changes                                                                                                                    |
|`outrider_zscore`     |calculated zscores                                                                                                                              |
|`outrider_delta`      |calculated delta values (obs - exp)                                                                                                             |
|`X_raw`               |input matrix (same as `adata.X`)                                                                                                                |
|`X_prepro`            |contains the result of applying the preprocessing function (in `adata.uns["prepro_func"]`) to `X_raw` (equal to `X_raw` if no preprocessing)    |
|`X_sf_norm`           |sizefactor normalization applied to `X_prepro`                                                                                                  |
|`X_transformed`       |contains the result of applying the transformation function (in `adata.uns["transform_func"]`) to `X_sf_norm` (or `X_prepro` when sf_norm=False)|
|`X_AE_input`          |input matrix to the AE model, including covariates if requested (`X_transformed` centered feature-wise if centering=True)                       |
|`X_predicted_no_trans`|output of AE model before reverse transforming (compare with `X_transformed`)                                                                   |
|`X_predicted`         |expected values, output of AE model after reverse transforming and multiplying with sizefactors (compare with `X_prepro`)                       |
|`X_latent`            |computed latent space (samples x encod_dim); access with `adata.obsm["X_latent"]`                                                               |
|`dispersions`         |calculated dispersions (= theta for NB); access with `adata.varm["dispersions"]`                                                                |


Additionally, py_outrider writes a results table to a specified csv file if the option `--output_res_tabe /PATH/TO/RESULTS_TABLE.CSV` is set. 

---
## Usage
```
usage: py_outrider [-h] [-in INPUT] [-sa SAMPLE_ANNO] [-o OUTPUT]
                   [-ot {h5ad,csv,zarr}] [-or OUTPUT_RES_TABLE]
                   [-p {outrider,protrider,pca}] [-q ENCOD_DIM]
                   [-cov [COVARIATES [COVARIATES ...]]] [-i ITERATIONS]
                   [-fl {float32,float64}] [-cpu NUM_CPUS] [-v VERBOSE]
                   [-s SEED] [-dis {NB,gaussian,log-gaussian}]
                   [-ld {NB,gaussian,log-gaussian}] [-pre {none,log}]
                   [-sf SF_NORM] [-c CENTERING] [-dt {log1p,log,none}]
                   [-nf NOISE_FACTOR] [--latent_space_model {AE,PCA}]
                   [--decoder_model {AE,PCA}] [--dispersion_model {ML,MoM}]
                   [--optimizer {lbfgs}] [--batch_size BATCH_SIZE]
                   [--nr_latent_space_features NR_LATENT_SPACE_FEATURES]
                   [--parallelize_D] [--no_parallelize_D]
                   [--convergence CONVERGENCE]
                   [--effect_type [{none,zscores,fold_change,delta} [{none,zscores,fold_change,delta} ...]]]
                   [--fdr_method FDR_METHOD] [--max_iter_hyper MAX_ITER_HYPER]
                   [--convergence_hyper CONVERGENCE_HYPER]

Run py_outrider to detect aberrant events in omics data.

optional arguments:
  -h, --help            show this help message and exit
  -in INPUT, --input INPUT
                        Path to a file containing the input data matrix, like
                        a gene count table. Can either be a csv file with
                        format: rows:samples x columns:genes and 1. row and 1.
                        column having feature and sample names, respectively;
                        or a .h5ad file containing an existing AnnData object
                        .
  -sa SAMPLE_ANNO, --sample_anno SAMPLE_ANNO
                        Path to sample annotation file, must be comma
                        separated, automatically selects sampleID column as
                        the one which has input_file rownames in it.
  -o OUTPUT, --output OUTPUT
                        Filename to save results (adata object).
  -ot {h5ad,csv,zarr}, --output_type {h5ad,csv,zarr}
                        File type of the output (h5ad, csv or zarr). Default:
                        h5ad
  -or OUTPUT_RES_TABLE, --output_res_table OUTPUT_RES_TABLE
                        Outputs results table as csv to the specifed file.
  -p {outrider,protrider,pca}, --profile {outrider,protrider,pca}
                        Choose which pre-defined implementation should be
                        used. Default profile: outrider
  -q ENCOD_DIM, --encod_dim ENCOD_DIM
                        Number of encoding dimensions. If not specified, runs
                        hyperparameter optimization to determine best encoding
                        dimension.
  -cov [COVARIATES [COVARIATES ...]], --covariates [COVARIATES [COVARIATES ...]]
                        List of covariate names in sample_anno, can either be
                        columns filled with 0 and 1, multiple numbers or
                        strings (will be automatically converted to one-hot
                        encoded matrix for training).
  -i ITERATIONS, --iterations ITERATIONS
                        [predefined in profile] Number of maximal training
                        iterations. Default: 15
  -fl {float32,float64}, --float_type {float32,float64}
                        [predefined in profile] Which float type should be
                        used, highly advised to keep float64 which may take
                        longer, but accuracy is strongly impaired by float32.
                        Default: float64
  -cpu NUM_CPUS, --num_cpus NUM_CPUS
                        Number of cpus used. Default: 1
  -v VERBOSE, --verbose VERBOSE
                        Set this flag to enable printing of additional
                        information during model fitting.
  -s SEED, --seed SEED  Seed used for training, negative values (e.g. -1) ->
                        no seed set. Default: 7
  -dis {NB,gaussian,log-gaussian}, --distribution {NB,gaussian,log-gaussian}
                        [predefined in profile] distribution assumed for the
                        measurement data.
  -ld {NB,gaussian,log-gaussian}, --loss_distribution {NB,gaussian,log-gaussian}
                        [predefined in profile] loss distribution used for
                        training.
  -pre {none,log}, --prepro_func {none,log}
                        [predefined in profile] preprocessing function applied
                        to input data. Distribution should be applicable for
                        the preprocessed data.
  -sf SF_NORM, --sf_norm SF_NORM
                        [predefined in profile] Boolean value indicating
                        whether sizefactor normalization should be performed.
  -c CENTERING, --centering CENTERING
                        [predefined in profile] Boolean value indicating
                        whether input should be centered before model fitting.
  -dt {log1p,log,none}, --data_trans {log1p,log,none}
                        [predefined in profile] transformation function
                        applied to preprocessed input data to create input for
                        AE model.
  -nf NOISE_FACTOR, --noise_factor NOISE_FACTOR
                        [predefined in profile] factor which defines the
                        amount of noise applied for a denoising autoencoder
                        model.
  --latent_space_model {AE,PCA}
                        [predefined in profile] Sets the model for fitting the
                        latent space.
  --decoder_model {AE,PCA}
                        [predefined in profile] Sets the model for fitting the
                        decoder.
  --dispersion_model {ML,MoM}
                        [predefined in profile] Sets the model for fitting the
                        dispersion parameters. Either ML for maximum
                        likelihood fit or MoM for methods of moments. Has no
                        effect for loss distributions that do not have
                        dispersion parameters.
  --optimizer {lbfgs}   [predefined in profile] Sets the optimizer for model
                        fitting. Currently only L-BFGS is implemented.
  --batch_size BATCH_SIZE
                        batch_size used for model fitting. Default is to use
                        all samples.
  --nr_latent_space_features NR_LATENT_SPACE_FEATURES
                        Limits the number of features used for fitting the
                        latent space to the specified number k. The k most
                        variable features will be selected. Default is to use
                        all features.
  --parallelize_D       If this flag is set, parallelizes fitting of decoder
                        per feature. Default: True (do parallelize by
                        feature).
  --no_parallelize_D    If this flag is set, decoder fit will not be
                        parallelized per feature. Default: do parallelize by
                        feature.
  --convergence CONVERGENCE
                        Sets the convergence limit. Default value is 1e-5.
  --effect_type [{none,zscores,fold_change,delta} [{none,zscores,fold_change,delta} ...]]
                        [predefined in profile] Chooses the type of effect
                        size that is calculated. Specifying multiple options
                        is possible.
  --fdr_method FDR_METHOD
                        Sets the fdr adjustment method. Must be one of the
                        methods from
                        statsmodels.stats.multitest.multipletests. Defaults to
                        fdr_by.
  --max_iter_hyper MAX_ITER_HYPER
                        Number of maximial training iterations during hyper
                        parameter optimization. Default: 15
  --convergence_hyper CONVERGENCE_HYPER
                        Convergence limit used during hyper parameter
                        optimization. Default: 1e-5
```


