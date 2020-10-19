import numpy as np

from py_outrider.dataset_handling.preprocess.prepro_abstract import Prepro_abstract
from py_outrider.dataset_handling.input_transform.trans_sf import Trans_sf



class Prepro_sf_log(Prepro_abstract):


    @staticmethod
    def get_prepro_x(xrds):
        counts = xrds["X"].values
        sf = Trans_sf.calc_size_factor(counts)
        return np.log((counts + 1) /  np.expand_dims(sf,1))


