import os
import cv2
import numpy as np
import tensorflow as tf
from IPython import embed

from lib.architecture.Model import Model
from lib.loader.RecordReader import RecordReader
from lib.tools.file import choose_one_from_dir, choose_model


def show(image):
    cv2.imshow('', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    image_size = (512, 768, 3) # input is grayscale x 3
    results_dir = '/home/petar/Projects/Shapes/models'

    model = Model(name='Model', M=2)

    experiment = choose_one_from_dir(results_dir)
    model_dir = os.path.join(experiment, 'model')
    model_path = choose_model(model_dir)
    model.load_weights(model_path)

    inputs = tf.keras.Input(shape=image_size, dtype=tf.float32)
    model._set_inputs(inputs)


    # reader = RecordReader(
    #     record_dir='/home/petar/Projects/Shapes/records',
    #     record_name='segmentator_full_gray_rect_oval_smooth',
    #     batch_size=1,
    #     shuffle_buffer=2,
    #     num_parallel_calls=2,
    #     num_parallel_reads=2,
    #     prefatch_buffer_size=5,
    #     count=1,
    #     image_size=image_size[:-1],
    # )

    # train_record = reader.read_record('train')
    # test_record  = reader.read_record('test')

    # for i, (name, image, mask) in enumerate(record):
        # print(f'Image: {i}', end='\r')

    image_np = cv2.imread('/home/petar/Projects/Shapes/presentations_png_mmseg 6/test/Screen Shot 2022-06-29 at 16.06.44.png', cv2.IMREAD_GRAYSCALE)
    image_np = cv2.resize(image_np, (768, 512))
    image_np = np.stack((image_np, image_np, image_np), axis=-1)

    show(image_np)

    image_np = np.expand_dims(image_np, axis=0)

    image_np = (image_np / 255).astype(np.float32)

    prediction = model(image_np)

    embed()

    # image = (image[0].numpy() * 255).astype(np.uint8)
    # show(image)
    # mask = mask[0].numpy()
    pred = prediction[0].numpy()

    # mask = np.concatenate([mask, np.zeros_like(mask)[..., 0:1]], axis=-1)
    pred = np.concatenate([pred, np.zeros_like(pred)[..., 0:1]], axis=-1)

    # mask = (mask * 255).astype(np.uint8)
    pred = (pred * 255).astype(np.uint8)

    # show(mask)
    show(pred)
