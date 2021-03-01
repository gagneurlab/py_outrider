import tensorflow as tf
from tensorflow import math as tfm
import pandas as pd
import numpy as np
import time

from ..utils import print_func as print_func


# class which keeps track of the training process
class Loss_list():

    def __init__(self, ):
        self.start_time = time.time()
        self.loss_summary = pd.DataFrame(columns=['step', 'step_name', 'loss',
                                                  'time', 'time_from_start',
                                                  'time_from_start_sec'])
        self.losses = []

    def add_loss(self, loss, print_text="", step_name=''):
        if tf.is_tensor(loss):
            loss = loss.numpy()
        self.losses.append(loss)
        curr_sec = round(time.time() - self.start_time)

        loss_row = pd.DataFrame([[len(self.losses), step_name, loss,
                                  time.strftime("%c"),
                                  print_func.get_duration_sec(curr_sec),
                                  curr_sec]],
                                columns=['step', 'step_name', 'loss', 'time',
                                         'time_from_start',
                                         'time_from_start_sec'])
        self.loss_summary = self.loss_summary.append(loss_row,
                                                     ignore_index=True)

        if print_text != "":
            FORCE_TEXT_LENGTH = 28
            print_text_filled = print_text + (FORCE_TEXT_LENGTH -
                                              len(print_text)) * " "
            print_func.print_time(print_text_filled + str(loss))

    def _is_converged(self, loss_list, conv_limit, last_iter, verbose):
        """
        check on current loss_list if it is converged and has not change a
        certain limit the last few steps
        :param loss_list: list of values (losses) or list of list of values,
        e.g. [[loss], [loss of 5 col]] to
        track convergence on a measurement level
        :param conv_limit: numeric convergence criteria
        :param last_iter: number on how many steps the convergence criteria
            must be satisfied to be accounted converged
        :param verbose: print additional information
        :return: tuple: has loss_list converged, are given loss_list[1] columns
            converged
        """
        if len(loss_list) > last_iter:
            l_short = loss_list[-(last_iter+1):-1]
            l_curr = loss_list[-1]
            loss_conv = np.abs(l_curr - l_short) < conv_limit
            meas_conv = np.all(loss_conv, axis=0)  # tf.reduce_all

            if np.size(loss_list[0]) > 1:
                num_converged = tfm.count_nonzero(meas_conv)
                if verbose:
                    cols_not_converged = len(meas_conv) - num_converged
                    print((
                        f'### cols_converged: {num_converged}',
                        f'     cols_not_converged: {cols_not_converged}'))
                return num_converged == len(meas_conv), str(
                    num_converged.numpy()) + '/' + str(
                    (len(meas_conv) - num_converged).numpy())
            else:
                conv = bool(meas_conv) is True
                if verbose:
                    print((f'### loss_converged: {meas_conv}'
                           f'   last {last_iter}+1 losses: '
                           f'{loss_list[-(last_iter+1):]}'))
                return conv, str(conv)
        return False, str(False)

    def check_converged(self, conv_limit, last_iter=3, verbose=False):
        return self._is_converged(self.losses, conv_limit, last_iter,
                                  verbose)[0]

    def save_loss_summary(self, file_path):
        self.loss_summary.to_csv(file_path, index=False)
