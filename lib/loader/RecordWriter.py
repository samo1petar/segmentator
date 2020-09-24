import cv2
import os
from random import shuffle
import tensorflow as tf
from typing import Dict, Generator, List, Tuple, Union
from lib.tools.progress_bar import printProgressBar


class RecordWriter:
    def __init__(
            self,
            data_path   : str = None,
            record_dir  : str = None,
            record_name : str = None,
            train_set   : Dict = None,
            test_set    : Dict = None,
            image_size  : Tuple[int, int] = (512, 512),
            save_n_test_images : int = None,
            save_n_train_images : int = None,
    ):
        assert os.path.exists(record_dir)
        assert record_name

        self._record_dir  = record_dir
        self._record_name = record_name
        self._image_size  = image_size

        self._images_path = os.path.join(data_path, 'images')
        self._masks_path = os.path.join(data_path, 'masked_images')

        self._train_record = os.path.join(self._record_dir, self._record_name + '_train' + '.tfrecord')
        self._test_record  = os.path.join(self._record_dir, self._record_name + '_test' + '.tfrecord')

        if not os.path.exists(self._test_record):
            self.create_record(test_set, self._test_record)

        if not os.path.exists(self._train_record):
            self.create_record(train_set, self._train_record)

    def get_next(self, dataset: Dict[str, Dict[str, str]]) -> Generator[List[Union[str, str, bytes]], None, None]:

        for key in dataset:

            name = key

            image = cv2.imread(dataset[key]['image'], cv2.IMREAD_COLOR)
            image = cv2.resize(image, (self._image_size[1], self._image_size[0]))
            image = cv2.imencode('.png', image)[1].tostring()

            mask = cv2.imread(dataset[key]['mask'], cv2.IMREAD_COLOR)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask[mask > 25] = 255
            mask[mask <= 25] = 0
            mask = cv2.resize(mask, (self._image_size[1], self._image_size[0]))
            mask = cv2.GaussianBlur(mask, (0, 0), cv2.BORDER_DEFAULT)
            mask = cv2.imencode('.png', mask)[1].tostring()

            yield name, image, mask

    def create_record(self, dataset: Dict[str, Dict[str, str]], full_record_name: str) -> None:

        print ('Creating record {}'.format(full_record_name))

        def write(
                name     : str,
                image    : bytes,
                mask     : bytes,
                writer   : tf.compat.v1.python_io.TFRecordWriter,
        ) -> None:
            feature = {
                'name'     : tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(name)])),
                'image'    : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'mask'     : tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        try:
            with tf.compat.v1.python_io.TFRecordWriter(full_record_name) as writer:
                count = 0
                num_iterate = len(dataset)
                for name, image, mask in self.get_next(dataset):
                    count += 1
                    write(name, image, mask, writer)
                    printProgressBar(count, num_iterate, decimals=1, length=50, suffix=' {} / {}'.format(count, num_iterate))

        except Exception as e:
            print ('Writing record failed, erasing record file {}'.format(full_record_name))
            print ('Erorr {}'.format(e))
            os.remove(full_record_name)
