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
    def ae_input_trans(self):
        return self.__ae_input_trans

    @ae_input_trans.setter
    def ae_input_trans(self, ae_input_trans):
        self.__ae_input_trans = ae_input_trans



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

    # @property
    # def loss_D(self):
    #     return self.__loss_D
    #
    # @loss_D.setter
    # def loss_D(self, loss_D):
    #     self.__loss_D = loss_D






    @property
    def noise_factor(self):
        return self.__noise_factor

    @noise_factor.setter
    def noise_factor(self, noise_factor):
        self.__noise_factor = noise_factor




    # def get_profile_str(self):




