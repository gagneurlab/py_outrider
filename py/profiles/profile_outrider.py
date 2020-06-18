###outrider and protrider


from profiles.profile_abstract import Profile_abstract
from ae_models.ae_pca import Ae_pca


class Profile_outrider(Profile_abstract):

    def __init__(self):
        # self.ae_model = bfgs
        # self.ae_input_norm = sf-norm
        # self.ae_input_norm_rev = sf-norm rev
        # self.distribution = neg_bin
        # self.loss_D = loss_d
        # self.loss_E = loss_e
        # self.noise_factor = 0

        self.ae_model = Ae_pca
        self.ae_input_norm = "sf"
        self.distribution = 'neg_bin'
        self.loss_D = 0
        self.loss_E = 0
        self.noise_factor = 0


