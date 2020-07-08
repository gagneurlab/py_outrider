import numpy as np
import utilis.stats_func as st
from sklearn.model_selection import ParameterGrid



class Hyperpar_opt():

    def __init__(self, ds):
        self.ds = ds

        ds.inject_outlier(inj_freq=1e-3, inj_mean=3, inj_sd=1.6)

        # xrds = fit_model.run_model_fit()
        # print(xrds)

        ### get all hyperparamters
        hyperpar_grid = self.get_hyperpar_grid( self.ds, encod_dim = True, noise_factor = False)

        # pre_rec = st.get_prec_recall(xrds["X_pvalue"].values, xrds["X_is_outlier"].values)["auc"]


        ### reverse injected outliers
        self.ds.xrds["X_inj_outlier"] = (('sample', 'meas'), self.ds.xrds["X"])
        self.ds.xrds["X"] = (('sample', 'meas'), self.ds.xrds["X_wo_outlier"])
        self.ds.xrds["X_trans"] = (('sample', 'meas'), self.ds.xrds["X_trans_wo_outlier"])

        self.apply_best_hyperpar()


    # def run_hyperpar_opt(self, encod_dim, noise_factor=None):
    #     fit_model = model_ds.profile.fit_model(model_ds)
    #     print('run model')
    #     xrds = fit_model.run_model_fit()


    # ### TODO list to include more hyperparameters:
    # ### learning_rate, minibatch_size (bad), # training_epochs, amount_noise
    # def run_hyperpar_opt(self, encod_dim):
    #
    #     for q in encod_dim:
    #         print_ext.print_time('####### start with encod_dim = '+str(q))
    #         time_start = time.time()
    #         self.ds_obj.encoding_dim = q
    #
    #         ae = self.ae_class(self.ds_obj)
    #         ae.run_autoencoder(max_iter=10)
    #
    #         counts_pred = np.around(ae.y_pred.numpy(), decimals=5)
    #         theta = np.around(ae.cov_meas.numpy() ,decimals=5)
    #
    #         np.savetxt(dh.path(self.folder_path, 'counts_pred_dim'+str(q)+'.csv'), counts_pred, delimiter=",")
    #         np.savetxt(dh.path(self.folder_path, 'theta_dim'+str(q)+'.csv'), theta, delimiter=",")
    #
    #         auc_pr = get_prec_recall(ae.pval, self.outlier_pos)['auc']
    #         print('### encoding_dim: {} with auc_pr: {}'.format(q, auc_pr))
    #         print_ext.print_time('### encoding_dim: {} with auc_pr: {}'.format(q, auc_pr))
    #
    #         time_end = print_ext.get_duration_sec(time.time() - time_start)
    #         with open(self.folder_path + 'hyperopt_stats.txt', 'a+') as f:
    #             f.write(str(q)+','+str(auc_pr)+','+str(ae.get_loss())+','+str(time_end)+'\n')





    def apply_best_hyperpar(self):
        self.ds.xrds.attrs["encod_dim"] = 5
        self.ds.profile.noise_factor = 2
        pass
        ## get best encod dim
        ## get best noise factor
        # hyper_df = pd.DataFrame(columns=['encod_dim','noise_factor', 'loss', 'pr_auc','time'])




    def get_hyperpar_grid(self, ds, encod_dim, noise_factor):
        hyperpar_dict = {}

        if encod_dim:
            hyperpar_dict["encod_dim"] = self._get_par_encod_dim(ds.xrds["X"])
        else:
            hyperpar_dict["encod_dim"] = [round(ds.xrds["X"].shape[0] / 4)]  # default 0.25 x samples

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
        return par_q.tolist()


    def _get_par_noise_factors(self):
        return [0, 0.5, 1, 1.5, 2]














