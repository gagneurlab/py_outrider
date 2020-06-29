###outrider and protrider


from profiles.profile_abstract import Profile_abstract
from ae_models.fitting_models.ae_bfgs2 import Ae_bfgs
from distributions.dis.dis_gaussian import Dis_gaussian
from distributions.dis.dis_log_gaussian import Dis_log_gaussian
from distributions.loss_dis.loss_dis_gaussian import Loss_dis_gaussian
from distributions.tf_loss_func import tf_gaus_loss_E, tf_gaus_loss_D_single

class Profile_protrider(Profile_abstract):

    def __init__(self):

        self.ae_model = Ae_bfgs
        # self.ae_model = Ae_pca
        self.ae_input_trans = "log2"
        self.dis = Dis_log_gaussian
        self.loss_dis = Loss_dis_gaussian
        self.noise_factor = 0.5



