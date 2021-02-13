
from .fit_components.fit_models import Model_fit_lbfgs, Model_fit_pca
from .distributions.dis_neg_bin import Dis_neg_bin
from .distributions.dis_log_gaussian import Dis_log_gaussian
from .distributions.dis_gaussian import Dis_gaussian
from .dataset_handling.input_transform import Trans_none, Trans_sf, Trans_log
from .dataset_handling.preprocess import Prepro_none, Prepro_sf_log


class Profile():

    def __init__(self, type):
        type = type.lower()
        assert type in ('outrider', 'protrider', 'pca'), '%s is not a valid profile type.' % type
        self.type = type
        
        # OUTRIDER profile settings
        if type == "outrider":
            self.fit_model = Model_fit_lbfgs
            self.prepro = Prepro_none
            self.data_trans = Trans_sf
            self.dis = Dis_neg_bin
            self.loss_dis = Dis_neg_bin
            self.outlier_dis = Dis_log_gaussian
            self.noise_dis = Dis_gaussian
            self.noise_factor = 0
        elif type == "protrider":
            self.fit_model = Model_fit_lbfgs
            self.prepro = Prepro_sf_log
            self.data_trans = Trans_none
            self.dis = Dis_gaussian
            self.loss_dis = Dis_gaussian
            self.outlier_dis = Dis_log_gaussian
            self.noise_dis = Dis_gaussian
            self.noise_factor = 0.5
        elif type == "pca":
            self.fit_model = Model_fit_pca
            self.prepro = Prepro_none
            self.data_trans = Trans_log
            self.dis = Dis_log_gaussian
            self.loss_dis = Dis_gaussian
            self.outlier_dis = Dis_log_gaussian
            self.noise_dis = Dis_gaussian
            self.noise_factor = 0

    @property
    def fit_model(self):
        return self.__fit_model

    @fit_model.setter
    def fit_model(self, fit_model):
        self.__fit_model = fit_model

    @property
    def prepro(self):
        return self.__prepro

    @prepro.setter
    def prepro(self, prepro):
        self.__prepro = prepro


    @property
    def data_trans(self):
        return self.__data_trans

    @data_trans.setter
    def data_trans(self, data_trans):
        self.__data_trans = data_trans



    @property
    def dis(self):
        return self.__dis

    @dis.setter
    def dis(self, dis):
        self.__dis = dis

    @property
    def loss_dis(self):
        return self.__loss_dis

    @loss_dis.setter
    def loss_dis(self, loss_dis):
        self.__loss_dis = loss_dis

    @property
    def outlier_dis(self):
        return self.__outlier_dis

    @outlier_dis.setter
    def outlier_dis(self, outlier_dis):
        self.__outlier_dis = outlier_dis

    @property
    def noise_dis(self):
        return self.__noise_dis

    @noise_dis.setter
    def noise_dis(self, noise_dis):
        self.__noise_dis = noise_dis


    @property
    def noise_factor(self):
        return self.__noise_factor

    @noise_factor.setter
    def noise_factor(self, noise_factor):
        self.__noise_factor = noise_factor


    def get_names(self):
        out_dict = {
            "profile_type": self.type,
            "fit_model": self.fit_model.__name__ ,
            "prepro" : self.prepro.__name__ ,
            "data_trans" : self.data_trans.__name__ ,
            "dis": self.dis.__name__ ,
            "loss_dis" : self.loss_dis.__name__ ,
            "outlier_dis" : self.outlier_dis.__name__ ,
            "noise_dis": self.noise_dis.__name__ ,
            "noise_factor": self.noise_factor }

        return out_dict
        
    def __str__(self):
        return str(self.__class__.__name__) + ": " + str(self.get_names()) 
        


