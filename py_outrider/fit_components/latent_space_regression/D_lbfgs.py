import tensorflow as tf    # 2.0.0
import tensorflow_probability as tfp

from .D_abstract import D_abstract


class D_lbfgs(D_abstract):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.ds.D is None:
            raise ValueError("D is none, need approximate weights for D to perform LBGFS refinement")


    def run_fit(self):
        if self.parallelize is True:
            D_optim_obj = self.get_updated_D_single(loss_func=self.loss_D,
                                            x=self.ds.X, H = self.ds.H, b=self.ds.b,
                                            D=self.ds.D,
                                            sizefactors=self.ds.sizefactors, dispersions=self.ds.dispersions,
                                            data_trans=self.ds.profile.data_trans,
                                            parallel_iterations=self.ds.parallel_iterations)
        else:
            D_optim_obj = self.get_updated_D_whole(loss_func=self.loss_D,
                                            x=self.ds.X, H = self.ds.H, b=self.ds.b,
                                            D=self.ds.D,
                                            sizefactors=self.ds.sizefactors, dispersions=self.ds.dispersions,
                                            data_trans=self.ds.profile.data_trans,
                                            parallel_iterations=self.ds.parallel_iterations)
            
        b = D_optim_obj["b_optim"]
        D = D_optim_obj["D_optim"]

        self._update_weights(D=D, b=b)



    def _update_weights(self, D, b):
        self.ds.D = tf.convert_to_tensor(D, dtype=self.ds.X.dtype)
        self.ds.b = tf.convert_to_tensor(b, dtype=self.ds.X.dtype)





    ### finds global minimum across whole matrix and not over columns - works as well
    @staticmethod
    @tf.function(experimental_relax_shapes=True)
    def get_updated_D_whole(loss_func, x, H, b, D, sizefactors, dispersions, data_trans, parallel_iterations=1):

        b_and_D = tf.concat([tf.expand_dims(b, 0), D], axis=0)
        b_and_D = tf.reshape(b_and_D, [-1])  ### flatten

        def lbfgs_input(b_and_D):
            loss = loss_func(H=H, x=x, b_and_D=b_and_D, data_trans=data_trans, dispersions=dispersions, sizefactors=sizefactors)
            gradients = tf.gradients(loss, b_and_D)[0]  ## works but runtime check, eager faster ??
            return loss, tf.clip_by_value(tf.reshape(gradients, [-1]), -100., 100.)

        optim = tfp.optimizer.lbfgs_minimize(lbfgs_input, initial_position=b_and_D, tolerance=1e-8, max_iterations=100, #300, #100,
                                             num_correction_pairs=10, parallel_iterations = parallel_iterations)
        b_and_D_out = tf.reshape(optim.position, [D.shape[0] + 1, D.shape[1]])

        return {"b_optim": b_and_D_out[0, :], "D_optim": b_and_D_out[1:, :], "optim": optim}


    @tf.function(experimental_relax_shapes=True)
    def get_updated_D_single(self, loss_func, x, H, b, D, sizefactors, dispersions, data_trans, parallel_iterations=1):
        meas_cols = tf.range(tf.shape(D)[1])
        map_D = tf.map_fn(lambda i: (self.single_fit_D(loss_func, H, x[:, i], b[i], D[:, i], sizefactors, dispersions[i], data_trans)),
                          meas_cols,
                          dtype=x.dtype,
                          parallel_iterations=parallel_iterations)
        return {"b_optim":map_D[:, 0], "D_optim":tf.transpose(map_D[:, 1:]) }  # returns b and D


    @staticmethod
    @tf.function
    def single_fit_D(loss_func, H, x_i, b_i, D_i, sizefactors, dispersions, data_trans):
        b_and_D = tf.concat([tf.expand_dims(b_i, 0), D_i], axis=0)

        def lbfgs_input(b_and_D):
            loss = loss_func(H=H, x_i=x_i, b_and_D=b_and_D, dispersions=dispersions, sizefactors=sizefactors, data_trans=data_trans)
            gradients = tf.gradients(loss, b_and_D)[0]
            return loss, tf.clip_by_value(gradients, -100., 100.)

        optim = tfp.optimizer.lbfgs_minimize(lbfgs_input, initial_position=b_and_D, tolerance=1e-6,
                                             max_iterations=50)
        return optim.position  # b(200) and D(200x10) -> needs t()
        
    # @tf.function
    # def get_updated_D_single(self, loss_func, x, H, b, D, sizefactors, dispersions, data_trans, parallel_iterations=1):
    #     
    #     @tf.function
    #     def fast_single_fit_D(tensors_to_vectorize):
    #         print("Tracing single D fit with tensors = ", tensors_to_vectorize)
    #         x_i, D_i, b_i, dispersions_i = tensors_to_vectorize
    #         b_and_D = tf.concat([tf.expand_dims(b_i, 0), D_i], axis=0)
    #         print("In single D fit: b_and_d = ", b_and_D)
    #         print("In single D fit: x_i = ", x_i)
    #         print("In single D fit: dispersions_i = ", dispersions_i)
    #     
    #         @tf.function
    #         def lbfgs_input(b_and_D):
    #             loss = loss_func(H=H, x_i=x_i, b_and_D=b_and_D, dispersions=dispersions_i, sizefactors=sizefactors, data_trans=data_trans)
    #             gradients = tf.gradients(loss, b_and_D)[0]
    #             return loss, tf.clip_by_value(gradients, -100., 100.)
    #         
    #         optim = tfp.optimizer.lbfgs_minimize(lbfgs_input, 
    #                                              initial_position=b_and_D, 
    #                                              tolerance=1e-6,
    #                                              max_iterations=50)
    #         return optim.position 
    # 
    #     map_D = tf.vectorized_map(fast_single_fit_D, (tf.transpose(x), tf.transpose(D), b, dispersions)) #tensors_to_vectorize = (x, D, b, dispersions)
    #     # print("Vectorized_map output = ", map_D)
    #     return {"b_optim":map_D[:, 0], "D_optim":tf.transpose(map_D[:, 1:]) }  # returns b and D









