###outrider and protrider


from profiles.profile_abstract import Profile_abstract
from ae_models.fitting_models.ae_bfgs2 import Ae_bfgs
from distributions.dis.dis_neg_bin import Dis_neg_bin
from distributions.tf_loss_func import tf_neg_bin_loss_E, tf_neg_bin_loss_D_single


class Profile_outrider(Profile_abstract):

    def __init__(self):

        # self.ae_model = Ae_pca
        self.ae_model = Ae_bfgs
        self.ae_input_trans = "sf"
        self.distribution = Dis_neg_bin
        self.loss_E = tf_neg_bin_loss_E
        self.loss_D = tf_neg_bin_loss_D_single
        self.noise_factor = 0


