
from fit_components.latent_space_fit.E_pca import E_pca
from fit_components.fitting_models.model_fit_abstract import Model_fit_abstract
from fit_components.par_meas_fit.par_meas_fminbound import Par_meas_fminbound




class Model_fit_pca(Model_fit_abstract):


    def __init__(self, ae_dataset):
        super().__init__(ae_dataset)

        self.ds.fit_input = self.ds.X_trans  # no covariate consideration in pca


    def run_fit(self, **kwargs):
        E_pca(ds=self.ds, **kwargs).run_fit()
        Par_meas_fminbound(ds=self.ds, **kwargs).run_fit()





