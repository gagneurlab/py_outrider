###outrider and protrider


from py_outrider.profiles.profile_abstract import Profile_abstract
from py_outrider.fit_components.fitting_models.model_fit_lbfgs import Model_fit_lbfgs
from py_outrider.fit_components.fitting_models.model_fit_pca import Model_fit_pca
from py_outrider.distributions.dis.dis_neg_bin import Dis_neg_bin
from py_outrider.distributions.dis.dis_log_gaussian import Dis_log_gaussian
from py_outrider.distributions.dis.dis_gaussian import Dis_gaussian
from py_outrider.distributions.loss_dis.loss_dis_neg_bin import Loss_dis_neg_bin
from py_outrider.distributions.loss_dis.loss_dis_log_gaussian import Loss_dis_log_gaussian
from py_outrider.distributions.loss_dis.loss_dis_gaussian import Loss_dis_gaussian
from py_outrider.dataset_handling.input_transform.trans_sf import Trans_sf
from py_outrider.dataset_handling.preprocess.prepro_none import Prepro_none
from py_outrider.dataset_handling.input_transform.trans_log import Trans_log
from py_outrider.dataset_handling.input_transform.trans_none import Trans_none



class Profile_outrider(Profile_abstract):

    def __init__(self):

        self.fit_model = Model_fit_lbfgs
        self.prepro = Prepro_none
        self.data_trans = Trans_sf
        self.dis = Dis_neg_bin
        self.loss_dis = Loss_dis_neg_bin
        self.outlier_dis = Dis_log_gaussian
        self.noise_dis = Dis_gaussian
        self.noise_factor = 0


