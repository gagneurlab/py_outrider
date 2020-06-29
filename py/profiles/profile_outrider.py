###outrider and protrider


from profiles.profile_abstract import Profile_abstract
from ae_models.fitting_models.model_fit_lbfgs import Model_fit_lbfgs
from distributions.dis.dis_neg_bin import Dis_neg_bin
from distributions.dis.dis_log_gaussian import Dis_log_gaussian
from distributions.loss_dis.loss_dis_neg_bin import Loss_dis_neg_bin
from dataset_handling.data_transform.trans_sf import Trans_sf



class Profile_outrider(Profile_abstract):

    def __init__(self):

        self.fit_model = Model_fit_lbfgs
        self.data_trans = Trans_sf
        self.dis = Dis_neg_bin
        self.loss_dis = Loss_dis_neg_bin
        self.outlier_dis = Dis_log_gaussian
        self.noise_factor = 0


