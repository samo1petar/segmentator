import tensorflow as tf


def optimizer_and_learning_rate(
        learning_rate : float,
        batch_size    : int,
        decay_steps   : int,
        decay_rate    : float,
) -> tf.keras.optimizers.Optimizer:

    initial_learning_rate = learning_rate * batch_size
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps = decay_steps,
        decay_rate  = decay_rate,
        staircase   = True)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    return tf.keras.optimizers.Adam(learning_rate=lr_schedule)
