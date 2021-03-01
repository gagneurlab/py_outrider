import tensorflow as tf


def init_tf_config(num_cpus, verbose):
    """
    prepares all configuration to make tensorflow run in parallel or use gpu if
    available
    :param num_cpus: number of maximum allowed cores
    :param verbose: print additional information output
    """
    tf.config.threading.set_intra_op_parallelism_threads(num_cpus)
    tf.config.threading.set_inter_op_parallelism_threads(num_cpus)

    physical_gpus = tf.config.experimental.list_physical_devices('GPU')

    if physical_gpus:
        print(f'Num of physical GPUs = {len(physical_gpus)} ({physical_gpus})')
        # Restrict TensorFlow to use only one GPU
        try:
            gpu = physical_gpus[0]
            print(f'Restricting TF to use only {gpu}')
            tf.config.experimental.set_visible_devices(gpu, 'GPU')
            print('Setting memory growth = True')
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print((
                f'Num of physical GPUs = {len(physical_gpus)}, '
                f'Num of logical GPUs = {len(logical_gpus)}'))
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    else:
        print('No physical GPUs detected')

    if verbose:
        print(f'allowed cpu_number: {num_cpus}')
        tf_inter_op = tf.config.threading.get_inter_op_parallelism_threads()
        tf_intra_op = tf.config.threading.get_intra_op_parallelism_threads()
        print(f'get_inter_op_parallelism_threads: {tf_inter_op}')
        print(f'get_intra_op_parallelism_threads: {tf_intra_op}')


def init_float_type(float_type):
    tf.keras.backend.set_floatx(float_type)


def init_tf_seed(seed):
    # sets only global seed and not operator seed
    # leads to identical values for tf.function calls only
    tf.random.set_seed(seed)
    # TODO check again if it really applies globally or lbfgs need additional
    # operator seed
