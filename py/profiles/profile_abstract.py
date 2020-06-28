from abc import ABC, abstractmethod




class Profile_abstract(ABC):

    def __init__(self):
        pass

    @property
    def ae_model(self):
        return self.__ae_model

    @ae_model.setter
    def ae_model(self, ae_model):
        self.__ae_model = ae_model


    @property
    def ae_input_norm(self):
        return self.__ae_input_norm

    @ae_input_norm.setter
    def ae_input_norm(self, ae_input_norm):
        self.__ae_input_norm = ae_input_norm



    @property
    def distribution(self):
        return self.__distribution

    @distribution.setter
    def distribution(self, distribution):
        self.__distribution = distribution

    @property
    def loss_D(self):
        return self.__loss_D

    @loss_D.setter
    def loss_D(self, loss_D):
        self.__loss_D = loss_D


    @property
    def loss_E(self):
        return self.__loss_E

    @loss_E.setter
    def loss_E(self, loss_E):
        self.__loss_E = loss_E


    @property
    def noise_factor(self):
        return self.__noise_factor

    @noise_factor.setter
    def noise_factor(self, noise_factor):
        self.__noise_factor = noise_factor




    # def get_profile_str(self):




