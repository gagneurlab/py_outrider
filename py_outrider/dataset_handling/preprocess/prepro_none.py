
from py_outrider.dataset_handling.preprocess.prepro_abstract import Prepro_abstract




class Prepro_none(Prepro_abstract):

    @staticmethod
    def get_prepro_x(xrds):
        return xrds["X"]





