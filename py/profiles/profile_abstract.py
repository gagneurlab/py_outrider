from abc import ABC, abstractmethod




class Profile_abstract(ABC):

    def __init__(self):
        pass

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






