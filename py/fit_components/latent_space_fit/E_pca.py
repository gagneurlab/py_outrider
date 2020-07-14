import numpy as np
import pandas as pd
import tensorflow as tf    # 2.0.0
from sklearn.decomposition import PCA
from nipals import nipals
# from autoencoder_models.loss_list import Loss_list

from fit_components.latent_space_fit.E_abstract import E_abstract


class E_pca(E_abstract):


    E_name = "E_PCA"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def run_fit(self):

        ### nipals if nan values are in matrix
        if np.isnan(self.ds.fit_input).any():
            pca_coef = self.get_weights_nipals(fit_input=self.ds.fit_input, encod_dim=self.ds.encod_dim)
        else:
            pca_coef = self.get_weights_pca(fit_input=self.ds.fit_input, encod_dim=self.ds.encod_dim)

        self._update_weights(pca_coef)




    def get_weights_nipals(self, fit_input, encod_dim):
        try:
            ### sometimes fails for big encoding dim for small samples
            nip = nipals.Nipals(fit_input)
            # nip.fit(ncomp=encod_dim, maxiter=500,tol=0.000001 ) # default
            nip.fit(ncomp=encod_dim, maxiter=1500,tol=0.00001 )
            return np.transpose(nip.loadings.to_numpy())
        except:
            print(f"INFO: nipals failed for encod_dim {encod_dim}, using imputed matrix and PCA")
            ### emergency solution -> fix otherway
            fit_df =  pd.DataFrame(fit_input)
            fit_input_imputed = fit_df.fillna(fit_df.mean()).to_numpy()
            return self.get_weights_pca(fit_input_imputed, encod_dim)



    def get_weights_pca(self, fit_input, encod_dim):
        pca = PCA(n_components=encod_dim, svd_solver='full')
        pca.fit(fit_input)
        return pca.components_  # encod_dim x samples





    def _update_weights(self, pca_coef):
        self.ds.E = tf.convert_to_tensor(np.transpose(pca_coef), dtype=self.ds.X.dtype)

        if self.ds.D is None:
            ### restructure for initialization with covariants
            if self.ds.cov_sample is not None:
                pca_coef_D = pca_coef[:, :-self.ds.cov_sample.shape[1]]
                cov_init_weights = np.zeros(shape=(self.ds.cov_sample.shape[1], pca_coef_D.shape[1]))
                pca_coef_D = np.concatenate([pca_coef_D, cov_init_weights], axis=0)
            else:
                pca_coef_D = pca_coef

            self.ds.D = tf.convert_to_tensor(pca_coef_D, dtype=self.ds.X.dtype)
            self.ds.b = tf.convert_to_tensor(self.ds.X_center_bias, dtype=self.ds.X.dtype)

        _, H = self.reshape_e_to_H(e=self.ds.E, fit_input=self.ds.fit_input_noise, X=self.ds.X, D=self.ds.D, cov_sample=self.ds.cov_sample)
        self.ds.H = H















