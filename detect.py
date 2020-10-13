import cv2
import os
import numpy as np
import tensorflow as tf
import time
from lib.tools.file import choose_one_from_dir, choose_model
from lib.loader.DataLoader import DataLoader
from IPython import embed
from lib.architecture.Model import Model as SegmentationModel
from lib.architecture.ModelCard import Model as CardModel
from lib.tools.softmax import softmax
import itertools
import matplotlib.pyplot as plt
from lib.detection.image import show, show_and_return_key
from lib.detection.thresholding import threshold_images_2
from lib.detection.image import dewarp


def get_segmentation_model(results_dir):
    print ('Choose segmentation model')
    model = SegmentationModel(name='Model', M=1)
    experiment = choose_one_from_dir(results_dir)
    model_dir = os.path.join(experiment, 'model')
    model_path = choose_model(model_dir)
    model.load_weights(model_path)
    inputs = tf.keras.Input(shape=(512, 512, 3), dtype=tf.float32)
    model._set_inputs(inputs)

    return model


def get_cards_model(results_dir):
    print ('Choose card classifier model')
    model = CardModel(name='Model', M=1)
    experiment = choose_one_from_dir(results_dir)
    model_dir = os.path.join(experiment, 'model')
    model_path = choose_model(model_dir)
    model.load_weights(model_path)
    inputs = tf.keras.Input(shape=(256, 256, 3), dtype=tf.float32)
    model._set_inputs(inputs)

    return model


def predict_segmentation(original_image, segmentation_model):
    original_image = cv2.resize(original_image, (512, 512))
    image = np.expand_dims(cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR), axis=0) / 255

    image_tf = tf.convert_to_tensor(image)

    prediction_tf = segmentation_model(image_tf)

    prediction_ = prediction_tf.numpy()[0]

    return prediction_


def predict_card(detection, card_model):
    cls_ids = {
        'bear': 0,
        'cat': 1,
        'cow': 2,
        'dog': 3,
        'elephant': 4,
        'giraffe': 5,
        'goat': 6,
        'horse': 7,
        'lion': 8,
        'monkey': 9,
        'pig': 10,
        'tiger': 11,
    }

    cls_names = {value: key for key, value in cls_ids.items()}

    input_image = cv2.resize(detection, (256, 256))
    input_image = np.expand_dims(cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR), axis=0) / 255

    image_tf = tf.convert_to_tensor(input_image)

    cls_pred = card_model(image_tf).numpy()[0]

    cls_pred = softmax(cls_pred)

    pred_cls = cls_names[np.argmax(cls_pred)]
    pred_perc = cls_pred[np.argmax(cls_pred)]

    return pred_cls, pred_perc



def detect_cards(original_image, verbose=False):

    def fillhole(input_image):
        '''
        input gray binary image  get the filled image by floodfill method
        Note: only holes surrounded in the connected regions will be filled.
        :param input_image:
        :return:
        '''
        im_flood_fill = input_image.copy()
        h, w = input_image.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        im_flood_fill = im_flood_fill.astype("uint8")
        cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
        im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
        img_out = input_image | im_flood_fill_inv
        return img_out

    def contour_approximation(image, original_image):
        cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        total = 0

        detections = []

        mask_with_contour = original_image.copy()
        new_image = np.zeros_like(image)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [c], [255, 255, 255])
            pixels = cv2.countNonZero(mask)
            total += pixels
            percentage = pixels / (image.shape[0] * image.shape[1])
            if percentage > 0.008:
                new_image[mask == 255] = 255

                epsilon = 0.04 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)

                if approx.shape[0] >= 4:
                    detections.append(approx)

                # print (approx.shape)

                mask_with_contour = cv2.drawContours(original_image.copy(), approx, -1, (0, 255, 255), 6)

                # print (mask_with_contour.shape)

                # show(mask_with_contour, title='CONTOUR')

        return new_image, mask_with_contour, detections

    def PolyArea(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def make_it_bigger(image):

        kernel = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ])

        filtered = cv2.filter2D(image, -1, kernel)

        return filtered

    def process(original_image, verbose):
        image = threshold_images_2(original_image)

        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(grayscale, 30, 100)

        if verbose:
            show(original_image, 'original_image')
            show(image, 'thresholded_image')
            show(edges)

        edges = make_it_bigger(edges)

        flooded_edges = fillhole(edges)

        if verbose:
            show(edges, 'Canny')
            show(flooded_edges, 'flood fill')

        mask, mask_with_contour, detections = contour_approximation(flooded_edges, original_image)

        return mask, mask_with_contour, detections

    mask, mask_with_contour, detections = process(original_image, verbose=verbose)

    if verbose:
        show(mask)
        show(mask_with_contour)

    rotated = False

    if len(detections) == 1 and PolyArea(detections[0][:, :, 0].reshape(-1),
                                         detections[0][:, :, 1].reshape(-1)) > 0.8 * (mask.shape[0] * mask.shape[1]):
        # print ('Rotate')
        original_image = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)
        mask, mask_with_contour, detections = process(original_image, verbose)

        rotated = True

    cards = []
    cards_detections = []

    for detection in detections:
        if detection.shape[0] == 4:
            det = detection.reshape(4, 2)

            height = np.linalg.norm(det[0] - det[2])
            width = np.linalg.norm(det[1] - det[3])

            # print ('height', height, 'width', width, 'distance', height - width)

            card_image_1 = dewarp(original_image, np.array([list(det[2]), list(det[1]), list(det[3]), list(det[0])]))
            card_image_2 = dewarp(original_image, np.array([list(det[3]), list(det[2]), list(det[0]), list(det[1])]))

            cards.append((card_image_1, card_image_2))
            cards_detections.append([list(det[2]), list(det[1]), list(det[3]), list(det[0])])

            # show(card_image_1)
            # show(card_image_2)

        if detection.shape[0] > 4:
            det = detection.reshape(-1, 2)

            max_size = 0
            max_det = None

            for set in itertools.combinations(det.tolist(), 4):
                x = np.array([x[0] for x in set])
                y = np.array([x[1] for x in set])

                area_size = PolyArea(x, y)

                if area_size > max_size:
                    max_size = area_size
                    max_det = np.hstack((x.reshape(4, 1), y.reshape(4, 1)))

            card_image_1 = dewarp(original_image,
                                  np.array([list(max_det[2]), list(max_det[1]), list(max_det[3]), list(max_det[0])]))
            card_image_2 = dewarp(original_image,
                                  np.array([list(max_det[3]), list(max_det[2]), list(max_det[0]), list(max_det[1])]))

            cards.append((card_image_1, card_image_2))
            cards_detections.append([list(max_det[2]), list(max_det[1]), list(max_det[3]), list(max_det[0])])

            # show(card_image_1)
            # show(card_image_2)

    return cards, cards_detections, rotated

def predict(
        results_dir : str,
        data_path   : str,
        name        : str,
) -> None:

    time_measurement = tf.keras.metrics.Mean(name='time_measurement')
    # time_measurement.reset_states()

    segmentation_model = get_segmentation_model(results_dir)
    cards_model = get_cards_model(results_dir)

    loader = DataLoader(data_path=data_path)

    for x in loader.yield_annotations_from_path(loader.get_data_as_list(shuffle=True)):
        path = x['path']
        cls = x['cls']
        im_name = x['im_name']
        input_device = x['input_device']

        # if not 'multiclass' in path:
        #     continue

        original_image = cv2.imread(path)

        if original_image.shape[1] > 900:
            original_image = cv2.resize(original_image, (900, int(900 * original_image.shape[0] / original_image.shape[1])))

        cards, cards_detections, rotated = detect_cards(original_image)

        if rotated and len(cards) > 0:
            original_image = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)

        cls = []
        for x in range(len(cards)):
            card_cls_1, score_1 = predict_card(cards[x][0], cards_model)
            card_cls_2, score_2 = predict_card(cards[x][1], cards_model)

            print ()
            if score_1 > score_2:
                print ('Card cls is ', card_cls_1, ' with score ', score_1)
                cls.append(card_cls_1)
            else:
                print('Card cls is ', card_cls_2, ' with score ', score_2)
                cls.append(card_cls_2)

            points = np.array(cards_detections[x]).reshape(4, 2)
            x = points[3].copy()
            points[3] = points[2]
            points[2] = x

            center_x = np.sum(points[:, 0]) / 4
            center_y = np.sum(points[:, 1]) / 4

            # original_image = cv2.circle(original_image, (center_x.astype(np.int32), center_y.astype(np.int32)), 5, 5)

            original_image = cv2.putText(original_image, cls[-1], (center_x.astype(np.int32), center_y.astype(np.int32)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

            original_image = cv2.polylines(original_image, points.reshape(-1, 4, 2), True, (0, 255, 255), 2)

        # show(original_image)

        play_pad_mask = predict_segmentation(original_image, segmentation_model)

        play_pad_mask = cv2.resize(play_pad_mask, (original_image.shape[1], original_image.shape[0]))
        play_pad_mask = (play_pad_mask * 255).astype(np.uint8)
        play_pad_mask = cv2.cvtColor(play_pad_mask, cv2.COLOR_GRAY2BGR)

        merged_image = (play_pad_mask * 0.5 + original_image * 0.5).astype(np.uint8)

        big_image = np.hstack((original_image, merged_image))
        show(big_image)

if __name__ == '__main__':
    predict(
        results_dir = '/home/david/Projects/Segmentator/models',
        data_path   = '/media/david/A/Datasets/PlayHippo/images',
        name='',
    )
