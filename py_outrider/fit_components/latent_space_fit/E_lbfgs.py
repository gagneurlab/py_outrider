import tensorflow as tf    # 2.0.0
import tensorflow_probability as tfp

# from autoencoder_models.loss_list import Loss_list

from py_outrider.fit_components.latent_space_fit.E_abstract import E_abstract



class E_lbfgs(E_abstract):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.ds.E is None:
            raise ValueError("E is none, need approximate weights for E to perform LBGFS refinement")


    def run_fit(self):
        E_optim_obj = self.get_updated_E(loss_func = self.loss_E,
                                         E = self.ds.E, D= self.ds.D, b = self.ds.b, x = self.ds.X, x_trans = self.ds.fit_input_noise,
                                         cov_sample=self.ds.cov_sample, par_sample = self.ds.par_sample, par_meas = self.ds.par_meas,
                                         data_trans=self.ds.profile.data_trans,
                                         parallel_iterations=self.ds.parallel_iterations)

        E, H = self.reshape_e_to_H(e=E_optim_obj["E_optim"], fit_input=self.ds.fit_input_noise,
                                   X=self.ds.X, D=self.ds.D, cov_sample=self.ds.cov_sample)
        self._update_weights(E=E, H=H)



    def _update_weights(self, E, H):
        self.ds.E = tf.convert_to_tensor(E, dtype=self.ds.X.dtype)
        self.ds.H = tf.convert_to_tensor(H, dtype=self.ds.X.dtype)




    @tf.function
    def get_updated_E(self, loss_func, E, D, b, x, x_trans, par_sample, par_meas, cov_sample, data_trans, parallel_iterations=1):
        e = tf.reshape(E, shape=[tf.size(E), ])

        def lbfgs_input(e):
            loss = loss_func(e=e, D=D, b=b, x=x, x_trans=x_trans, par_sample=par_sample,
                             par_meas=par_meas, cov_sample=cov_sample, data_trans=data_trans)
            gradients = tf.gradients(loss, e)[0]
            return loss, tf.clip_by_value(gradients, -100., 100.)

        optim = tfp.optimizer.lbfgs_minimize(lbfgs_input, initial_position=e, tolerance=1e-8, max_iterations=100, #500, #150,
                                             num_correction_pairs=10, parallel_iterations=parallel_iterations)

        return {"E_optim":optim.position, "optim":optim }    # e and optimizer





