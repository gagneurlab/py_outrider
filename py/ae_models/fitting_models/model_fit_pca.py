
from ae_models.encoder_fit.E_pca import E_pca
from ae_models.fitting_models.model_fit_abstract import Ae_abstract
from ae_models.par_meas_fit.par_meas_fminbound import Par_meas_fminbound




class Ae_pca(Ae_abstract):


    def __init__(self, ae_dataset):
        super().__init__(ae_dataset)

        self.ds.ae_input = self.ds.X_trans  # no covariate consideration in pca


    def run_fit(self, **kwargs):
        E_pca(self.ds).run_fit()
        Par_meas_fminbound(self.ds).run_fit()





