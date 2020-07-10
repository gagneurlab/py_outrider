import numpy as np
import utilis.stats_func as st
from sklearn.model_selection import ParameterGrid
from utilis import print_func
import pandas as pd



class Hyperpar_opt():

    def __init__(self, ds):
        self.ds = ds

        ds.inject_outlier(inj_freq=1e-3, inj_mean=3, inj_sd=1.6)

        ### get all hyperparamters
        hyperpar_grid = self.get_hyperpar_grid( self.ds, encod_dim = True, noise_factor = False)  # noise factor TRUE

        hyperpar_table = self.run_hyperpar_opt(self.ds, hyperpar_grid)

        ### reverse injected outliers
        self.ds.xrds["X_inj_outlier"] = (('sample', 'meas'), self.ds.xrds["X"])
        self.ds._remove_inj_outlier()

        self.apply_best_hyperpar(hyperpar_table)



    def run_hyperpar_opt(self, ds, par_grid):
        print_func.print_time("start hyperparameter optimisation")

        df_list = []
        for p in par_grid:
            print(p)

            ### set hyperpar
            ds.encod_dim = p["encod_dim"]
            ds.profile.noise_factor = p["noise_factor"]

            ds.inject_noise(inj_freq=1, inj_mean=0, inj_sd=1)

            fit_model = ds.profile.fit_model(ds)
            fit_model.fit()
            ds.calc_pvalue()

            pre_rec = st.get_prec_recall(ds.X_pvalue, ds.xrds["X_is_outlier"].values)["auc"]
            df_list.append([p["encod_dim"], p["noise_factor"], pre_rec, ds.get_loss()])

        hyperpar_df = pd.DataFrame(df_list, columns=["encod_dim","noise_factor","prec_rec","loss"])

        print_func.print_time("end hyperparameter optimisation")
        return hyperpar_df




    def apply_best_hyperpar(self, df):
        self.ds.xrds.attrs["hyperpar_table"] = df.to_dict("records")

        best_row = df.loc[df['prec_rec'].idxmax()]
        self.ds.encod_dim = best_row["encod_dim"]
        self.ds.profile.noise_factor = best_row["noise_factor"]

        print('best hyperparameter found:')
        print(best_row)
        print(df)





    def get_hyperpar_grid(self, ds, encod_dim, noise_factor):
        hyperpar_dict = {}

        if encod_dim:
            hyperpar_dict["encod_dim"] = self._get_par_encod_dim(ds.xrds["X"])
        else:
            q = round(ds.xrds["X"].shape[0] / 4) if ds.encod_dim is None else ds.encod_dim
            hyperpar_dict["encod_dim"] = [q]

        if noise_factor:
            hyperpar_dict["noise_factor"] = self._get_par_noise_factors()
        else:
            hyperpar_dict["noise_factor"] = [ds.profile.noise_factor]

        return ParameterGrid(hyperpar_dict)


    ### from DROP
    def _get_par_encod_dim(self, x):
        MP = 3

        a = 5
        b = round(min(x.shape) / MP)
        max_steps = 15

        n_steps = min(max_steps, b)  # do at most 15 steps or N/3
        par_q = np.unique(np.round(np.exp(np.linspace(start=np.log(a), stop=np.log(b), num=n_steps))))
        return par_q.astype(int).tolist()


    def _get_par_noise_factors(self):
        return [0, 0.5, 1, 1.5, 2]














