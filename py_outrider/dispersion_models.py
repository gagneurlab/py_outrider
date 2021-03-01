import tensorflow as tf

from .utils.tf_fminbound import tf_fminbound


# Maximum Likelihood fit of dispersion parameters
class Dispersions_ML():

    def __init__(self, distribution):
        self.dispersions = None
        self.distribution = distribution
        self.loss = distribution.loss

    def get_dispersions(self):
        if self.dispersions is None:
            return None
        else:
            return self.dispersions.numpy()

    def init(self, x_true):
        mom = Dispersions_MoM(self.distribution)
        mom.init(x_true)
        self.dispersions = mom.dispersions

    @tf.function
    def lbfgs_input(self, disp, x_true, x_pred, min=1e-2, max=1e3):
        # disp = tf.minimum(tf.maximum(disp, min), max)
        disp = tf.Variable(disp)
        with tf.GradientTape() as tape:
            loss = self.loss(dispersions=disp, x_true=x_true, x_pred=x_pred)
        # gradients = tf.gradients(loss, disp)[0]
        gradients = tape.gradient(loss, disp)
        # print(f"Disp Loss: {loss}")
        # print(f"Disp Gradients: {gradients}")
        # return loss, tf.clip_by_value(gradients, -100., 100.)
        return loss, gradients

    @tf.function
    def _fit_fmbinbound(self, x_true, x_pred, n_parallel, loss):

        def my_map(*args, **kwargs):
            return tf.map_fn(*args, **kwargs)

        x_true_pred_stacked = tf.transpose(tf.stack([x_true, x_pred], axis=1))
        new_disp = my_map(
            lambda row: tf_fminbound(
                lambda t: loss(x_true=row[0, :], x_pred=row[1, :],
                               dispersions=t),
                x1=tf.constant(1e-2),
                x2=tf.constant(1e3)),
            x_true_pred_stacked, parallel_iterations=n_parallel)

        # TODO vectorized_map implementation
        # @tf.function
        # def optimize_loss_feature(t_row):
        #     return tf_fminbound(lambda disp: loss(dispersions=disp,
        #                                           x_true=t_row[0],
        #                                           x_pred=t_row[1]),
        #                         x1=tf.constant(1e-2),
        #                         x2=tf.constant(1e3))
        #
        # t = [tf.transpose(x_true), tf.transpose(x_pred)]
        # # print(t)
        # new_disp = tf.vectorized_map(optimize_loss_feature, t)

        # print(f"fmbinbound dispersions: {new_disp}")

        return new_disp

    def fit(self, x_true, x_pred, optimizer, n_parallel):
        if optimizer == "fminbound":
            self.dispersions = self._fit_fmbinbound(x_true, x_pred, n_parallel,
                                                    self.loss)

        return self.loss(dispersions=self.dispersions, x_true=x_true,
                         x_pred=x_pred)


# Method of moment fit for dispersions
class Dispersions_MoM():

    def __init__(self, distribution):
        self.dispersions = None
        self.distribution = distribution

    def get_dispersions(self):
        if self.dispersions is None:
            return None
        else:
            return self.dispersions.numpy()

    def init(self, x_true):
        # method of moments
        self.dispersions = self.distribution.mom(x_true)
        self.dispersions = tf.convert_to_tensor(self.dispersions)

    def fit(self, **kwargs):
        pass


DISPERSION_MODELS = {'ML': Dispersions_ML, 'MoM': Dispersions_MoM}
