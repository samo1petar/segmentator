import cv2
import os
import numpy as np
import tensorflow as tf
import time
from lib.tools.file import choose_one_from_dir, choose_model
from lib.loader.DataLoader import DataLoader
from IPython import embed
from lib.architecture.Model import Model


def show(image):
    cv2.imshow('', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def predict(
        model       : tf.keras.Model,
        results_dir : str,
        name        : str,
) -> None:

    time_measurement = tf.keras.metrics.Mean(name='time_measurement')
    # time_measurement.reset_states()

    experiment = choose_one_from_dir(results_dir)

    model_dir = os.path.join(experiment, 'model')
    model_path = choose_model(model_dir)

    model.load_weights(model_path)

    inputs = tf.keras.Input(shape=(512, 512, 3), dtype=tf.float32)
    model._set_inputs(inputs)

    loader = DataLoader(data_path='/media/david/A/Datasets/PlayHippo/images')

    for x in loader.yield_annotations_from_path(loader.get_data_as_list(shuffle=True)):
        path = x['path']
        cls = x['cls']
        im_name = x['im_name']
        input_device = x['input_device']

        original_image = cv2.imread(path)
        original_image = cv2.resize(original_image, (512, 512))
        image = np.expand_dims(cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR), axis=0) / 255

        start = time.time()

        image_tf = tf.convert_to_tensor(image)

        prediction_tf = model(image_tf)

        end = time.time()
        time_measurement(end - start)

        print (end - start)

        prediction_ = prediction_tf.numpy()[0]
        prediction_rgb = cv2.cvtColor(prediction_, cv2.COLOR_GRAY2RGB)
        merge_images = np.hstack((original_image, (prediction_rgb * 255).astype(np.uint8)))
        show(merge_images)

if __name__ == '__main__':
    predict(
        model = Model(name='Model', M=1),
        results_dir = '/home/david/Projects/Segmentator/results',
        name='',
    )
