import numpy as np

from dataset_handling.preprocess.prepro_abstract import Prepro_abstract
from dataset_handling.input_transform.trans_sf import Trans_sf



class Prepro_sf_log(Prepro_abstract):

    prepro_name = "prepro_sf_log"

    @staticmethod
    def get_prepro_x(xrds):
        counts = xrds["X"].values
        sf = Trans_sf.calc_size_factor(counts)
        return np.log((counts + 1) /  np.expand_dims(sf,1))


