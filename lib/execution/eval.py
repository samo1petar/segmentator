import os
import tensorflow as tf
import numpy as np
from lib.loader.classes import classes_encode, classes_decode
from lib.loader.RecordReader import RecordReader
from lib.tools.file import choose_one_from_dir, choose_model, mkdir
from lib.tools.softmax import softmax
from lib.tools.plot import save_figure


def eval(
        model       : tf.keras.Model,
        loader      : RecordReader,
        results_dir : str,
        save_images : bool,
) -> None:

    experiment = choose_one_from_dir(results_dir)
    eval_train_dir = os.path.join(experiment, 'eval_train')
    eval_test_dir = os.path.join(experiment, 'eval_test')
    mkdir(eval_train_dir)
    mkdir(eval_test_dir)
    model_dir = os.path.join(experiment, 'model')
    model_path = choose_model(model_dir)

    model.load_weights(model_path)

    inputs = tf.keras.Input(shape=(512, 512, 3), dtype=tf.float32)
    model._set_inputs(inputs)
    model.save(os.path.join(experiment, 'test_model'))

    loader._count = 1
    loader._batch_size = 1

    def process_set(
            source               : str,
            save_dir             : str,
            save_images          : bool = True,
            max_save_images      : int = 200,
            max_processed_images : int = 600,
    ):
        assert source in ['train', 'test']
        print ('Evaluating {}...'.format(source))
        conf_matrix = np.zeros((len(classes_encode), len(classes_encode)))
        br = 0
        wrong_images_count = 0
        for name, cls, cls_name, image in loader.read_record(source):
            br += 1
            if br > max_processed_images:
                break
            print (' [ {} ]'.format(br), end='\r')
            prediction = model(image, training=False)
            prediction = softmax(prediction.numpy())
            pred_cls = np.argmax(prediction)
            label_cls = np.argmax(cls.numpy())
            conf_matrix[pred_cls][label_cls] += 1
            if pred_cls != label_cls and save_images:
                wrong_images_count += 1
                if wrong_images_count < max_save_images:
                    save_figure(
                        image       = image.numpy()[0],
                        gt          = cls.numpy()[0],
                        pred        = prediction[0],
                        name        = name.numpy()[0].decode('utf8'),
                        destination = os.path.join(save_dir, name.numpy()[0].decode('utf8') + '.png')
                    )
        print ('{} conf_matrix\n{}'.format(source, conf_matrix))
        F1_list = []
        for x in range(len(classes_decode)):
            P = conf_matrix[x][x] / np.sum(conf_matrix, axis=1)[x]
            R = conf_matrix[x][x] / np.sum(conf_matrix, axis=0)[x]
            F1 = 2 * P * R / (P + R)
            F1_list.append(F1)
            print('{} Precision {} Recall {} F1 {}'.format(
                classes_decode[x],
                P,
                R,
                F1,
            ))
        print ('Macro F1 {}'.format(sum(F1_list) / len(F1_list)))
        TP = np.sum([conf_matrix[x][x] for x in range(len(classes_decode))])
        print ('Accuracy {}'.format(TP / np.sum(conf_matrix)))

    process_set('test', eval_test_dir, save_images)
    process_set('train', eval_train_dir, save_images)
