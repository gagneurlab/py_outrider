###outrider and protrider


from profiles.profile_abstract import Profile_abstract
from ae_models.fitting_models.ae_bfgs2 import Ae_bfgs
from distributions.dis.dis_neg_bin import Dis_neg_bin
from distributions.loss_dis.loss_dis_neg_bin_short import Loss_dis_neg_bin_short
from distributions.tf_loss_func import tf_neg_bin_loss_E, tf_neg_bin_loss_D_single


class Profile_outrider(Profile_abstract):

    def __init__(self):

        # self.ae_model = Ae_pca
        self.ae_model = Ae_bfgs
        self.ae_input_trans = "sf"
        self.dis = Dis_neg_bin
        self.loss_dis = Loss_dis_neg_bin_short
        self.noise_factor = 0


