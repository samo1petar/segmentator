import cv2
import os
from random import shuffle
import tensorflow as tf
from typing import Generator, List, Tuple, Union
from lib.tools.progress_bar import printProgressBar


# TODO -> rewrite for detection

class RecordWriter:
    def __init__(
            self,
            data_path   : str = None,
            record_dir  : str = None,
            record_name : str = None,
            image_size  : Tuple[int, int] = (512, 512),
    ):
        assert os.path.exists(record_dir)
        assert record_name

        self._record_dir  = record_dir
        self._record_name = record_name
        self._image_size  = image_size

        self._train_record = os.path.join(self._record_dir, self._record_name + '_train' + '.tfrecord')
        self._test_record  = os.path.join(self._record_dir, self._record_name + '_test' + '.tfrecord')

        if not os.path.exists(self._test_record):
            test_images = [os.path.join(root, x) for root, subdir, files in os.walk(data_path) for x in files if '/test' in root]
            shuffle(test_images)
            self.create_record(test_images, self._test_record)

        if not os.path.exists(self._train_record):
            train_images = [os.path.join(root, x) for root, subdir, files in os.walk(data_path) for x in files if '/train' in root]
            shuffle(train_images)
            self.create_record(train_images, self._train_record)

    def get_next(self, images: List[str]) -> Generator[List[Union[str, str, bytes]], None, None]:

        for image_path in images:

            im_name = image_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
            country = image_path.rsplit('/', 3)[1]
            cls = classes_encode[country]
            cls_name = country
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (self._image_size[1], self._image_size[0]))
            image = cv2.imencode('.png', image)[1].tostring()

            yield im_name, cls, cls_name, image

    def create_record(self, images: List[str], full_record_name: str) -> None:

        print ('Creating record {}'.format(full_record_name))

        def write(
                name     : str,
                cls      : int,
                cls_name : str,
                image    : bytes,
                writer   : tf.compat.v1.python_io.TFRecordWriter,
        ) -> None:
            feature = {
                'name'     : tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(name)])),
                'cls'      : tf.train.Feature(int64_list=tf.train.Int64List(value=[cls])),
                'cls_name' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(cls_name)])),
                'image'    : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        try:
            with tf.compat.v1.python_io.TFRecordWriter(full_record_name) as writer:
                count = 0
                num_iterate = len(images)
                for name, cls, cls_name, image in self.get_next(images):
                    count += 1
                    write(name, cls, cls_name, image, writer)
                    printProgressBar(count, num_iterate, decimals=1, length=50, suffix=' {} / {}'.format(count, num_iterate))

        except Exception as e:
            print ('Writing record failed, erasing record file {}'.format(full_record_name))
            print ('Erorr {}'.format(e))
            os.remove(full_record_name)
