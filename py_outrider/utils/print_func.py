import time
import tensorflow as tf
from tensorflow import math as tfm
import numpy as np


def print_time(text, tf_print=False):
    if tf_print:
        tf.print('### ' + time.strftime("%c") + '   ' + text)
    else:
        print('### ' + time.strftime("%c") + '   ' + text, flush=True)


def print_dict(d):
    for i in d:
        print(i, ': ', d[i])


def get_duration_sec(sec):
    return time.strftime('%H:%M:%S', time.gmtime(sec))


def print_lbfgs_optimizer(opt, tf_print=True):
    """
    prints summary of optimizer parameters
    :param opt: tf.lbfgs_minimize output object
    :param tf_print: True for tf.print(), False for simple print()
    """
    if tf_print:
        print_func = tf.print
    else:
        print_func = print

    tf.print(opt[0:6])
    # TODO find some way to actually print the values... .numpy() does not work
    print_func('# LBFGS optimizer:   '
               'converged: {}'.format(opt.converged))
    print_func('#                    '
               'failed: {}'.format(opt.failed))
    print_func('#                    '
               'num_iterations: {}'.format(opt.num_iterations))
    print_func('#                    '
               'num_objective_evaluations: {}'.format(
                   opt.num_objective_evaluations))

    print_func('#                    '
               'objective_value: {}'.format(opt.objective_value))
    print_func('#                    '
               'position_deltas shape: {}'.format(opt.position_deltas.shape))
    print_func('#                    '
               'objective_gradient ' + print_tensor_summary(
                   opt.objective_gradient, return_only=True))
    print_func('#                    '
               'position_deltas ' + print_tensor_summary(
                   opt.position_deltas, return_only=True))
    print_func('#                    '
               'gradient_deltas ' + print_tensor_summary(
                   opt.gradient_deltas, return_only=True))


def print_tensor_summary(t, return_only=False):
    t_min = tfm.reduce_min(t)
    t_max = tfm.reduce_max(t)
    t_mean = tfm.reduce_mean(t)
    t_nan = tfm.count_nonzero(~tfm.is_finite(t))
    txt = 'min / max / mean / nan: {} / {} / {} / {} '.format(t_min, t_max,
                                                              t_mean, t_nan)
    if return_only:
        return txt
    else:
        tf.print(txt)


def np_summary(a):
    DEC_NUM = 3
    mi = round(np.nanmin(a), DEC_NUM)
    ma = round(np.nanmax(a), DEC_NUM)
    me = round(np.nanmedian(a), DEC_NUM)
    return f'min: {mi}, max: {ma}, median: {me}'
