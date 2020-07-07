
from dataset_handling.preprocess.prepro_abstract import Prepro_abstract




class Prepro_none(Prepro_abstract):

    prepro_name = "prepro_none"

    @staticmethod
    def get_prepro_x(xrds):
        return xrds["X"]





