###outrider and protrider


from profiles.profile_abstract import Profile_abstract
from ae_models.ae_pca import Ae_pca
from distributions.dis_neg_bin import Dis_neg_bin
from distributions.tf_loss_func import tf_neg_bin_loss


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
        self.distribution = Dis_neg_bin
        self.loss_D = tf_neg_bin_loss
        self.loss_E = tf_neg_bin_loss
        self.noise_factor = 0


