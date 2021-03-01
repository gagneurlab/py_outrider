"""Python backend package for OUTRIDER2 R/Bioconductor package."""

__version__ = "0.1.0"

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
# tf.autograph.set_verbosity(2)
