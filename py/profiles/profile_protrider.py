###outrider and protrider


from profiles.profile_abstract import Profile_abstract
from fit_components.fitting_models.model_fit_lbfgs import Model_fit_lbfgs
from fit_components.fitting_models.model_fit_pca import Model_fit_pca
from distributions.dis.dis_gaussian import Dis_gaussian
from distributions.dis.dis_log_gaussian import Dis_log_gaussian
from distributions.loss_dis.loss_dis_gaussian import Loss_dis_gaussian
from distributions.loss_dis.loss_dis_log_gaussian import Loss_dis_log_gaussian
from dataset_handling.input_transform.trans_log import Trans_log
from dataset_handling.input_transform.trans_sf import Trans_sf
from dataset_handling.input_transform.trans_none import Trans_none
from dataset_handling.preprocess.prepro_sf_log import Prepro_sf_log




class Profile_protrider(Profile_abstract):

    def __init__(self):

        self.fit_model = Model_fit_lbfgs
        # self.fit_model = Model_fit_pca
        self.prepro = Prepro_sf_log
        self.data_trans = Trans_none
        # self.data_trans = Trans_sf
        self.dis = Dis_gaussian
        self.loss_dis = Loss_dis_gaussian
        self.outlier_dis = Dis_log_gaussian
        self.noise_dis = Dis_gaussian
        self.noise_factor = 0.5 #0.5



