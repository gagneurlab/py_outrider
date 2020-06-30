import tensorflow as tf    # 2.0.0
import tensorflow_probability as tfp

# from autoencoder_models.loss_list import Loss_list

from fit_components.latent_space_regression.D_abstract import D_abstract


class D_lbfgs_whole(D_abstract):

    D_name = "D_LBFGS_whole"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.ds.D is None:
            raise ValueError("D is none, need approximate weights for D to perform LBGFS refinement")


    # @tf.function
    def fit(self):
        D_optim_obj = self.get_updated_D(loss_func=self.loss_D,
                                            x=self.ds.X, H = self.ds.H, b=self.ds.b,
                                            D=self.ds.D,
                                            par_sample=self.ds.par_sample, par_meas=self.ds.par_meas,
                                            data_trans=self.ds.profile.data_trans,
                                            parallel_iterations=self.ds.parallel_iterations)
        b = D_optim_obj["b_optim"]
        D = D_optim_obj["D_optim"]

        self._update_weights(D=D, b=b)



    def _update_weights(self, D, b):
        self.ds.D = tf.convert_to_tensor(D, dtype=self.ds.X.dtype)
        self.ds.b = tf.convert_to_tensor(b, dtype=self.ds.X.dtype)





    ### finds global minimum across whole matrix and not over columns - works as well
    @tf.function
    def get_updated_D(self, loss_func, x, H, b, D, par_sample, par_meas, data_trans, parallel_iterations=1):

        b_and_D = tf.concat([tf.expand_dims(b, 0), D], axis=0)
        b_and_D = tf.reshape(b_and_D, [-1])  ### flatten

        def lbfgs_input(b_and_D):
            loss = loss_func(H=H, x=x, b_and_D=b_and_D, data_trans=data_trans, par_meas=par_meas, par_sample=par_sample)
            gradients = tf.gradients(loss, b_and_D)[0]  ## works but runtime check, eager faster ??
            return loss, tf.clip_by_value(tf.reshape(gradients, [-1]), -100., 100.)

        optim = tfp.optimizer.lbfgs_minimize(lbfgs_input, initial_position=b_and_D, tolerance=1e-8, max_iterations=100, #300, #100,
                                             num_correction_pairs=10, parallel_iterations = parallel_iterations)
        b_and_D_out = tf.reshape(optim.position, [D.shape[0] + 1, D.shape[1]])

        return {"b_optim": b_and_D_out[0, :], "D_optim": b_and_D_out[1:, :], "optim": optim}










