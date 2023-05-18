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


def iou_coef(y_true, y_pred, smooth=1, threshold=0.5):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    y_true[y_true > threshold] = 1
    y_true[y_true <= threshold] = 0
    y_pred[y_pred > threshold] = 1
    y_pred[y_pred <= threshold] = 0
    intersection = np.sum(np.abs(y_true * y_pred), axis=(1,2,3))
    union = np.sum(y_true,(1,2,3))+np.sum(y_pred,(1,2,3))-intersection
    iou = np.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def accuracy(y_true, y_pred, diff_thr=0.1):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()

    diff = np.abs(y_true - y_pred)

    error_pixels = np.where(diff > diff_thr)[0].shape[0]

    return 1 - error_pixels / (y_pred.shape[0] * y_pred.shape[1] * y_pred.shape[2] * y_pred.shape[3])


def run_metrics(record, model, verbose=False):
    iou_scores = []
    acc_scores = []

    for i, (name, image, mask) in enumerate(record):
        print(f'Image: {i}', end='\r')

        prediction = model(image)

        score = iou_coef(mask, prediction)
        iou_scores.append(score)
        acc = accuracy(mask, prediction)
        acc_scores.append(acc)

        if verbose:
            print (score)
            print(name)

            # image = (image[0].numpy() * 255).astype(np.uint8)
            # show(image)
            mask = mask[0].numpy()
            pred = prediction[0].numpy()

            mask = np.concatenate([mask, np.zeros_like(mask)[..., 0:1]], axis=-1)
            pred = np.concatenate([pred, np.zeros_like(pred)[..., 0:1]], axis=-1)

            mask = (mask * 255).astype(np.uint8)
            pred = (pred * 255).astype(np.uint8)

            show(mask)
            show(pred)

    print(f'IoU score: {sum(iou_scores) / len(iou_scores)}')
    print(f'Acc score: {sum(acc_scores) / len(acc_scores)}')



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


    reader = RecordReader(
        record_dir='/home/petar/Projects/Shapes/records',
        record_name='segmentator_full_gray_rect_oval_smooth',
        batch_size=1,
        shuffle_buffer=2,
        num_parallel_calls=2,
        num_parallel_reads=2,
        prefatch_buffer_size=5,
        count=1,
        image_size=image_size[:-1],
    )

    train_record = reader.read_record('train')
    test_record  = reader.read_record('test')

    print('\nTest')
    run_metrics(test_record, model)
    print('\nTrain')
    run_metrics(train_record, model)
