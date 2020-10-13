import cv2
import os
import numpy as np
import tensorflow as tf
import time
from IPython import embed
from lib.tools.file import choose_one_from_dir, choose_model
from lib.loader.RecordReaderCards import RecordReaderCards
from lib.tools.softmax import softmax
from lib.architecture.ModelCard import Model
from lib.tools.plot import save_figure, show_figure


def show(image):
    cv2.imshow('', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def predict(
        model       : tf.keras.Model,
        results_dir : str,
        name        : str,
        save_dir    : str = '',
        show        : bool = False,
) -> None:

    time_measurement = tf.keras.metrics.Mean(name='time_measurement')
    # time_measurement.reset_states()

    experiment = choose_one_from_dir(results_dir)

    model_dir = os.path.join(experiment, 'model')
    model_path = choose_model(model_dir)

    model.load_weights(model_path)

    inputs = tf.keras.Input(shape=(256, 256, 3), dtype=tf.float32)
    model._set_inputs(inputs)

    reader = RecordReaderCards(
        record_dir='records',
        record_name='cards',
        batch_size=1,
        shuffle_buffer=1,
        num_parallel_calls=1,
        num_parallel_reads=1,
        prefatch_buffer_size=1,
        count=-1,
        image_size=(256, 256),
    )

    iterator_train = reader.read_record('test')

    for name, cls_name, cls_id, image in iterator_train:

        prediction = model(image, training=True)

        prediction = prediction.numpy()
        name = name.numpy()
        cls_name = cls_name.numpy()
        cls_id = cls_id.numpy()
        image = image.numpy()

        for x in range(len(cls_name)):

            name_ = name[x].decode('utf8')
            cls_name_ = cls_name[x]
            cls_id_ = cls_id[x]

            cls_array_ = np.zeros([12])
            cls_array_[cls_id_] = 1

            prediction_ = softmax(prediction[x])
            image_ = image[x]

            if show:
                show_figure(image_, cls_array_, prediction_)

            if save_dir:
                save_figure(image_, cls_array_, prediction_, name_,
                            os.path.join(save_dir, '_'.join(name_.rsplit('/')[-3:])).replace('.jpg', '.png'))


if __name__ == '__main__':
    predict(
        model = Model(name='Model', M=1),
        results_dir = '/home/david/Projects/Segmentator/results',
        name='',
        save_dir='',
        show=True,
    )
