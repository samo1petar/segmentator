import os
import tensorflow as tf
import time
from lib.loader.RecordReader import RecordReader
from lib.tools.TrainSupport import TrainSupport


def train(
        model            : tf.keras.Model,
        loader           : RecordReader,
        loss_object      : tf.keras.losses.Loss,
        optimizer        : tf.keras.optimizers.Optimizer,
        print_every_iter : int,
        eval_every_iter  : int,
        max_iter         : int,
        results_dir      : str,
        name             : str,
        clip_gradients   : float,
) -> None:

    train_support = TrainSupport(save_dir=results_dir, name=name)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_print_loss = tf.keras.metrics.Mean(name='train_print_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    time_measurement = tf.keras.metrics.Mean(name='time_measurement')

    iterator_train = loader.read_record('train')

    logdir = train_support._save_dir
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    @tf.function
    def train_step(input, labels):
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            prediction = model(input, training=True)
            loss = loss_object(labels, prediction)

        gradients = tape.gradient(loss, model.trainable_variables)

        # clipped_gradients = []
        # for grad in gradients:
        #     clipped_gradients.append(tf.clip_by_value(grad, -clip_gradients, clip_gradients))

        # from IPython import embed
        # embed()

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_print_loss(loss)
        train_loss(loss)

    train_loss.reset_states()
    train_print_loss.reset_states()
    test_loss.reset_states()

    iter = tf.constant(0, dtype=tf.int64)
    while iter < max_iter:
        iter += 1

        # #######################  EVAL  #######################
        if iter % eval_every_iter == 0:
            tf.summary.scalar('train_loss', train_loss.result(), iter)
            train_loss.reset_states()

            model.save_weights(os.path.join(train_support.model_saving_dir, 'model_' + str(iter.numpy())), save_format='tf')

            train_support.sample_from(model, iterator_train, train_support.sample_train_dir)
            train_support.sample_from(model, loader.read_record('test'), train_support.sample_test_dir)

            for name, image, mask in loader.read_record('test'):
                prediction = model(image, training=False)
                loss = loss_object(mask, prediction)
                test_loss(loss)
            tf.summary.scalar('test_loss', test_loss.result(), iter)
            test_loss.reset_states()

        ########################  ITER  ########################
        if iter % print_every_iter == 0:
            print('Iter: {} \tLoss: {} \tTime: {}'.format(iter, train_print_loss.result(), time_measurement.result()))
            train_print_loss.reset_states()
            time_measurement.reset_states()

        ######################  TRAIN STEP ######################
        name, image, mask = iterator_train.__next__()
        start = time.time()
        train_step(image, mask)
        end = time.time()
        time_measurement(end - start)
