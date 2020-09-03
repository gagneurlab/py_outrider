import numpy as np



def np_set_seed(fun, seed=None):
    def wrapped(*args, **kwargs):
        np.random.seed(seed)
        return fun(*args, **kwargs)
    return wrapped

















