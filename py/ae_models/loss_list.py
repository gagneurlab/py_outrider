import tensorflow as tf    # 2.0.0
from tensorflow import math as tfm
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
import time
import utilis.print_func as print_func



### class which keeps track of the training process

class Loss_list():

    def __init__(self, conv_limit=1e-5, last_iter=3):
        self.start_time = time.time()
        self.conv_limit = conv_limit
        self.last_iter = last_iter
        self.loss_summary = pd.DataFrame(columns=['step', 'step_name', 'loss',
                                                  'time', 'time_from_start', 'time_from_start_sec'])
        self.losses = []


    def add_loss(self, loss, print_text="", step_name=''):
        self.losses.append(loss)
        curr_sec = round(time.time() - self.start_time)

        loss_row = pd.DataFrame([[len(self.losses), step_name, loss, time.strftime("%c"),
                                  print_func.get_duration_sec(curr_sec), curr_sec ]],
                                columns=['step', 'step_name', 'loss',
                                         'time', 'time_from_start', 'time_from_start_sec'])
        self.loss_summary = self.loss_summary.append(loss_row, ignore_index=True)
        if print_text is not "":
            FORCE_TEXT_LENGTH = 25
            print_text_filled = print_text + (FORCE_TEXT_LENGTH-len(print_text)) * " "
            print_func.print_time(print_text_filled + str(loss) )



    def _is_converged(self, loss_list, conv_limit, last_iter, verbose):
        if len(loss_list) > last_iter:
            l_short = loss_list[len(loss_list) - (last_iter + 1): len(loss_list) - 1]
            l_curr = loss_list[-1]
            loss_conv = np.abs(l_curr-l_short) < conv_limit
            meas_conv = tf.reduce_all(loss_conv, axis=0)

            if tf.size(loss_list[0]) > 1:
                num_converged = tfm.count_nonzero(meas_conv)
                if verbose:
                    tf.print('### genes_converged: {}     genes_not_converged: {}'.format(num_converged,
                                                                                           len(meas_conv) - num_converged))
                return num_converged == len(meas_conv), str(num_converged.numpy())+'/'+str((len(meas_conv) - num_converged).numpy())
            else:
                conv = meas_conv is True
                if verbose:
                    tf.print('### loss_converged: {}'.format(conv))
                return conv, str(conv)
        return False, str(False)



    def check_converged(self, verbose=False):
        self._is_converged(self.losses, self.conv_limit, self.last_iter, verbose)[0]


    def save_loss_summary(self, file_path):
        self.loss_summary.to_csv(file_path, index=False)


