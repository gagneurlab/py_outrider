import numpy as np
import utilis.stats_func as st




class Hyperpar_opt():

    def __init__(self, ds):
        self.ds = ds

        ds.inject_outlier(inj_freq=1e-3, inj_mean=3, inj_sd=1.6)
        
        xrds = fit_model.run_model_fit()
        print(xrds)


        encod_dim = np.arange(10, 50, 5)  # TODO automatically find appropriate values
        pre_rec = st.get_prec_recall(xrds["X_pvalue"].values, xrds["X_is_outlier"].values)["auc"]


        ### reverse injected outliers
        self.ds.xrds["X_inj_outlier"] = (('sample', 'meas'), self.ds.xrds["X"])
        self.ds.xrds["X"] = (('sample', 'meas'), self.ds.xrds["X_wo_outlier"])
        self.ds.xrds["X_trans"] = (('sample', 'meas'), self.ds.xrds["X_trans_wo_outlier"])
        self.ds.initialize_ds()




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
        pass
        ## get best encod dim
        ## get best noise factor
















