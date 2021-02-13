import time
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf    # 2.0.0
import tensorflow_probability as tfp
from sklearn.decomposition import PCA
import time
import math

from . import tf_init
from ..utils import print_func as print_func
from .latent_space_fit.E_pca import E_pca
from .latent_space_fit.E_lbfgs import E_lbfgs
from .latent_space_regression.D_lbfgs import D_lbfgs
from .dispersion_fit import Dispersions_fminbound, Dispersions_mom


class Model_fit_abstract(ABC):

    def __init__(self, model_dataset):
        self.ds = model_dataset
        self.ds.initialize_ds()

        tf_init.init_tf_config(num_cpus=self.ds.xrds.attrs["num_cpus"], verbose=self.ds.xrds.attrs["verbose"])
        tf_init.init_float_type(float_type=self.ds.xrds.attrs["float_type"])
        tf_init.init_tf_seed(seed=self.ds.xrds.attrs["seed"])


    @property
    def ds(self):
        return self.__ds

    @ds.setter
    def ds(self, ds):
        self.__ds = ds

    @abstractmethod
    def fit(self):
         pass

    def run_model_fit(self, **kwargs):
        time_ae_start = time.time()
        print_func.print_time('start model fitting')
        self.fit(**kwargs)
        print_func.print_time(f'complete model fit time {print_func.get_duration_sec(time.time() - time_ae_start)}')
        print_func.print_time(f'model_fit ended with loss: {self.ds.get_loss()}')

        self.ds.init_pvalue_fc_z()
        self.xrds = self.ds.get_xrds()
        return self.xrds



class Model_fit_pca(Model_fit_abstract):


    def __init__(self, ae_dataset):
        super().__init__(ae_dataset)



    def fit(self, **kwargs):
        E_pca(ds=self.ds, **kwargs).fit()
        Dispersions_fminbound(ds=self.ds, **kwargs).fit()
        
        

class Model_fit_lbfgs(Model_fit_abstract):


    def __init__(self, ae_dataset):
        super().__init__(ae_dataset)
        
        # check batch size
        batch_size = self.ds.xrds.attrs["batch_size"]
        max_batch_size = len(self.ds.xrds.coords["sample"])
        if batch_size is None:
            batch_size = max_batch_size
        assert batch_size > 0, "batch_size needs to be a positive number"
        self.batch_size = batch_size #if batch_size < max_batch_size else max_batch_size
        
        # check parallel fitting of D
        self.fit_D_parallel = self.ds.xrds.attrs["parallelize_D"] 
        if self.fit_D_parallel is None:
            self.fit_D_parallel = False

    @staticmethod
    def batch_split(ds, batch_size):
        nr_samples = len(ds.xrds.coords["sample"])
        sample_idx = tf.range(nr_samples)
        sample_idx = tf.random.shuffle(sample_idx)
        nr_batches = tf.constant(math.ceil(nr_samples / batch_size))
        batch_split = list()
        for b in tf.range(nr_batches):
            start_idx = b*batch_size
            end_idx = (b+1)*batch_size
            batch_split.append(sample_idx[start_idx:end_idx])
        return batch_split

    # @tf.function
    def fit(self, conv_limit=1e-5, **kwargs):

        print(f'### Initializing ...')
        # init E and D with pca
        E_pca(ds=self.ds).fit()

        # init theta with method of moment
        Dispersions_mom(ds=self.ds).fit()

        # inital D fit
        D_lbfgs(ds=self.ds, parallelize=self.fit_D_parallel).fit()

        # initial theta fit
        Dispersions_fminbound(ds=self.ds).fit()

        ### ITERATE UNTIL CONVERGENCE
        for iter in range(self.ds.xrds.attrs["max_iter"]):
            print(f'### ITERATION {iter+1}')
            time_iter_start = time.time()
            
            # get batches
            batches = self.batch_split(self.ds, self.batch_size)
            # for batch_id in range(len(batches)):
            #     if len(batches) > 1:
            #         print(f'### ### BATCH {batch_id+1}/{len(batches)}')
            #         ds_batch = self.ds.subset_samples(batches[batch_id])
            #     else:
            #         ds_batch = self.ds
            #     
            #     # Encoder fit
            #     E_lbfgs(ds=ds_batch).fit()
            #     # ds_batch.print_dataset_shapes()
            # 
            #     # Decoder fit
            #     D_lbfgs(ds=ds_batch, parallelize=self.fit_D_parallel).fit()
            # 
            #     # Dispersion fit
            #     Dispersions_fminbound(ds=ds_batch).fit()
            #     
            #     # update matrices in full dataset
            #     self.ds.E = ds_batch.E
            #     self.ds.D = ds_batch.D
            #     self.ds.b = ds_batch.b
            #     self.ds.dispersions = ds_batch.dispersions
            
            # Encoder fit
            print(f'### # Fitting the latent space ...')
            for batch_id in range(len(batches)):
                print(f'###\tBATCH {batch_id+1}/{len(batches)}: ', end='')
                if len(batches) > 1:
                    ds_batch = self.ds.subset_samples(batches[batch_id])
                else:
                    ds_batch = self.ds
                
                E_lbfgs(ds=ds_batch).fit()
                # update E in full dataset
                self.ds.E = ds_batch.E
            
            # Decoder fit
            print(f'### # Fitting the decoder ...')
            for batch_id in range(len(batches)):
                print(f'###\tBATCH {batch_id+1}/{len(batches)}: ', end='')
                if len(batches) > 1:
                    ds_batch = self.ds.subset_samples(batches[batch_id])
                else:
                    ds_batch = self.ds
    
                # ds_batch.print_dataset_shapes()
                D_lbfgs(ds=ds_batch, parallelize=self.fit_D_parallel).fit()
                # update D+b in full dataset
                self.ds.D = ds_batch.D
                self.ds.b = ds_batch.b
    
            # Dispersion fit
            print(f'### # Fitting the dispersion parameters ...')
            print(f'###\tBATCH 1/1: ', end='')
            Dispersions_fminbound(ds=self.ds).fit()

            print('duration loop: {}'.format(print_func.get_duration_sec(time.time() - time_iter_start)))

            ## check convergence
            if self.ds.loss_list.check_converged(conv_limit=conv_limit, verbose=self.ds.xrds.attrs["verbose"],
                                                 last_iter=len(batches)*2+1):
                print_func.print_time(f'model converged at iteration: {iter+1}')
                break









