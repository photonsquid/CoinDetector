import tensorflow as tf

# Avoid OOM errors by setting GPU Memory Consumption Growth


def set_gpus_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
