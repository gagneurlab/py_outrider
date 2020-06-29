import tensorflow as tf    # 2.0.0
import tensorflow_probability as tfp

# from autoencoder_models.loss_list import Loss_list

from ae_models.decoder_fit import D_abstract


class D_lbfgs_single(D_abstract):

    def __init__(self, **kwargs):
        self.__init__(**kwargs)
        self.loss_D = self.ds.profile.loss_dis.tf_loss_D_single

        if self.ds.D is None:
            raise ValueError("D is none, need aproximate weights for D to perform LBGFS refinement")



    def fit(self):
        D_optim_obj = self.get_updated_D(loss_func=self.loss_D,
                                       x=self.ds.X, H = self.ds.H, b=self.ds.b,
                                       D=self.ds.D,
                                       par_sample=self.ds.par_sample, par_meas=self.ds.par_meas,
                                       parallel_iterations=self.ds.parallel_iterations)
        b = D_optim_obj["b_optim"]
        D = D_optim_obj["D_optim"]

        self._update_weights(D=D, b=b)



    def _update_weights(self, D, b):
        self.ds.D = tf.convert_to_tensor(D, dtype=self.ds.X.dtype)
        self.ds.b = tf.convert_to_tensor(b, dtype=self.ds.X.dtype)



    @tf.function
    def get_updated_D(self, loss_func, x, H, b, D, par_sample, par_meas, parallel_iterations=1):
        meas_cols = tf.range(tf.shape(D)[1])
        map_D = tf.map_fn(lambda i: (self.single_fit_D(loss_func, H, x[:, i], b[i], D[:, i], par_sample, par_meas[i])),
                          meas_cols,
                          dtype=x.dtype,
                          parallel_iterations=parallel_iterations)
        return {"b_optim":map_D[:, 0], "D_optim":tf.transpose(map_D[:, 1:]) }  # returns b and D


    @tf.function
    def single_fit_D(self, loss_func, H, x_i, b_i, D_i, par_sample, par_meas_i):
        b_and_D = tf.concat([tf.expand_dims(b_i, 0), D_i], axis=0)

        def lbfgs_input(b_and_D):
            loss = loss_func(H, x_i, b_and_D, par_sample, par_meas_i)
            gradients = tf.gradients(loss, b_and_D)[0]
            return loss, tf.clip_by_value(gradients, -100., 100.)

        optim = tfp.optimizer.lbfgs_minimize(lbfgs_input, initial_position=b_and_D, tolerance=1e-6,
                                             max_iterations=50)
        return optim.position  # b(200) and D(200x10) -> needs t()






