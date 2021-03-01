import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from .utils import stats_func as st
from .utils import print_func
from .outrider import outrider
from .preprocess import inject_outliers


class Hyperpar_opt():

    def __init__(self, adata, **kwargs):
        self.hyperpar_table = None
        self.best_encod_dim = None
        self.best_noise_factor = None
        del kwargs['encod_dim']
        del kwargs['noise_factor']

        # inject artifical outliers
        self.adata = inject_outliers(adata, inj_freq=1e-3, inj_mean=3,
                                     inj_sd=1.6, **kwargs)
        kwargs["prepro_func"] = "none"

        # get all hyperparamters
        hyperpar_grid = self.get_hyperpar_grid(self.adata, encod_dim=True,
                                               noise_factor=True)
        self.hyperpar_table = self.run_hyperpar_opt(self.adata, hyperpar_grid,
                                                    **kwargs)
        self.apply_best_hyperpar(self.hyperpar_table)

    def run_hyperpar_opt(self, adata, par_grid, **kwargs):
        print_func.print_time("start hyperparameter optimisation")

        df_list = []
        for p in par_grid:
            print(p)

            # set hyperpar
            encod_dim = int(p["encod_dim"])
            noise_factor = p["noise_factor"]

            # hyperpar_data = adata.copy()
            res = outrider(adata,
                           encod_dim=encod_dim,
                           noise_factor=noise_factor,
                           **kwargs)

            pre_rec = st.get_prec_recall(res.layers["X_pvalue"],
                                         res.layers["X_is_outlier"])["auc"]
            df_list.append([encod_dim, noise_factor, pre_rec])

        hyperpar_df = pd.DataFrame(
                        df_list,
                        columns=["encod_dim", "noise_factor", "prec_rec"])

        print_func.print_time("end hyperparameter optimisation")
        return hyperpar_df

    def apply_best_hyperpar(self, df):
        self.hyperpar_table = df.to_dict("records")

        best_row = df.loc[df['prec_rec'].idxmax()]
        self.best_encod_dim = int(best_row["encod_dim"])
        self.best_noise_factor = best_row["noise_factor"]

        print_func.print_time('best hyperparameter found:')
        print(best_row)
        print_func.print_time('full hyperparameter optimization results:')
        print(df)

    def get_hyperpar_grid(self, adata, encod_dim, noise_factor):
        hyperpar_dict = {}

        if encod_dim:
            hyperpar_dict["encod_dim"] = self._get_par_encod_dim(adata.X)
        else:
            q = round(adata.X.shape[0] / 4)
            hyperpar_dict["encod_dim"] = [q]

        if noise_factor:
            hyperpar_dict["noise_factor"] = self._get_par_noise_factors()
        else:
            hyperpar_dict["noise_factor"] = [0.0]

        return ParameterGrid(hyperpar_dict)

    def _get_par_encod_dim(self, x):
        """
        get max_steps for hyperparameter optimisation for encoding dimension,
        log scattered for bigger measurement tables, from DROP
        :param x: measurement matrix
        :return: list of encoding dimensions
        """
        MP = 3

        a = 3
        b = round(min(x.shape) / MP)
        max_steps = 15

        n_steps = min(max_steps, b)  # do at most 15 steps or N/3
        par_q = np.unique(np.round(np.exp(np.linspace(start=np.log(a),
                                                      stop=np.log(b),
                                                      num=n_steps))))
        return par_q.astype(int).tolist()

    def _get_par_noise_factors(self):
        return [0, 0.5, 1]  # 1.5, 2
