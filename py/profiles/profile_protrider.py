###outrider and protrider


from profiles.profile_abstract import Profile_abstract


class Profile_protrider(Profile_abstract):

    def __init__(self):
        # self.ae_model = bfgs
        # self.ae_input_norm = np.log
        # self.ae_input_norm = np.exp
        # self.distribution = gaussian
        # self.loss_D = rmse
        # self.loss_E = rmse
        # self.noise_factor = 0

        self.ae_model = 0
        self.ae_input_norm = "log2"
        self.distribution = 0
        self.loss_D = 0
        self.loss_E = 0
        self.noise_factor = 0


