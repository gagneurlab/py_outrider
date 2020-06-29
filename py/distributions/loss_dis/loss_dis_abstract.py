from abc import ABC, abstractmethod



class Loss_dis_abstract(ABC):



    @staticmethod
    @abstractmethod
    def tf_loss(self):
         pass

    @staticmethod
    @abstractmethod
    def tf_loss_D(self):
         pass

    @staticmethod
    @abstractmethod
    def tf_loss_D_single(self):
         pass

    @staticmethod
    @abstractmethod
    def tf_loss_E(self):
        pass










