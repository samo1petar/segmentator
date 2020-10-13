import cv2
import os
from random import shuffle
import tensorflow as tf
from typing import Dict, Generator, List, Tuple, Union
from lib.tools.progress_bar import printProgressBar
from IPython import embed


cls_ids = {
    'bear'    : 0,
    'cat'     : 1,
    'cow'     : 2,
    'dog'     : 3,
    'elephant': 4,
    'giraffe' : 5,
    'goat'    : 6,
    'horse'   : 7,
    'lion'    : 8,
    'monkey'  : 9,
    'pig'     : 10,
    'tiger'   : 11,
}


class RecordWriterCards:
    def __init__(
            self,
            data_path   : str = None,
            record_dir  : str = None,
            record_name : str = None,
            image_size  : Tuple[int, int] = (512, 512),
            save_n_test_images : int = None,
            save_n_train_images : int = None,
    ):
        assert os.path.exists(record_dir)
        assert record_name

        self._record_dir  = record_dir
        self._record_name = record_name
        self._image_size  = image_size

        self._images_path = os.path.join(data_path, 'cards')

        self._train_record = os.path.join(self._record_dir, self._record_name + '_train' + '.tfrecord')
        self._test_record  = os.path.join(self._record_dir, self._record_name + '_test' + '.tfrecord')

        dataset = []
        for root, subdirs, files in os.walk(self._images_path):
            for file in files:
                dataset.append(os.path.join(root, file))

        shuffle(dataset)

        train_set = dataset[:int(0.8 * len(dataset))][:10]
        test_set = dataset[int(0.8 * len(dataset)):][:10]

        if not os.path.exists(self._test_record):
            self.create_record(test_set, self._test_record)

        if not os.path.exists(self._train_record):
            self.create_record(train_set, self._train_record)

    def get_next(self, dataset: Dict[str, Dict[str, str]]) -> Generator[List[Union[str, str, bytes]], None, None]:

        for path in dataset:

            cls_name = path.rsplit('/', 2)[1]
            cls_id = cls_ids[cls_name]

            image = cv2.imread(path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (self._image_size[1], self._image_size[0]))
            image = cv2.imencode('.png', image)[1].tostring()

            yield path, cls_name, cls_id, image

    def create_record(self, dataset: List[str], full_record_name: str) -> None:

        print ('Creating record {}'.format(full_record_name))

        def write(
                path     : str,
                cls_name : str,
                cls_id   : int,
                image    : bytes,
                writer   : tf.compat.v1.python_io.TFRecordWriter,
        ) -> None:
            feature = {
                'path'  : tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(path)])),
                'cls_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(cls_name)])),
                'cls_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[cls_id])),
                'image' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        try:
            with tf.compat.v1.python_io.TFRecordWriter(full_record_name) as writer:
                count = 0
                num_iterate = len(dataset)
                for path, cls_name, cls_id, image in self.get_next(dataset):
                    count += 1
                    write(path, cls_name, cls_id, image, writer)
                    printProgressBar(count, num_iterate, decimals=1, length=50, suffix=' {} / {}'.format(count, num_iterate))

        except Exception as e:
            print ('Writing record failed, erasing record file {}'.format(full_record_name))
            print ('Erorr {}'.format(e))
            os.remove(full_record_name)
