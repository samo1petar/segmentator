import os
import tensorflow as tf
from typing import Generator, Tuple


# TODO -> rewrite for detection

class RecordReader:
    def __init__(
            self,
            record_dir           : str = None,
            record_name          : str = None,
            image_size           : Tuple[int, int] = (512, 512),
            batch_size           : int = 1,
            shuffle_buffer       : int = 100,
            num_parallel_calls   : int = 8,
            num_parallel_reads   : int = 32,
            prefatch_buffer_size : int = 100,
            count                : int = None,
    ):
        assert os.path.exists(record_dir)
        assert record_name

        self._record_dir           = record_dir
        self._record_name          = record_name
        self._image_size           = image_size
        self._batch_size           = batch_size
        self._shuffle_buffer       = shuffle_buffer
        self._num_parallel_calls   = num_parallel_calls
        self._num_parallel_reads   = num_parallel_reads
        self._prefatch_buffer_size = prefatch_buffer_size
        self._count                = count

        self.record_path = os.path.exists(os.path.join(self._record_dir, self._record_name + '.tfrecord'))

    def read_record(self, name: str) -> Generator[tf.Tensor, None, None]:

        assert name in ['train', 'test']

        full_record_name = os.path.join(self._record_dir, self._record_name + '_' + name + '.tfrecord')

        def parse(x):
            keys_to_features = {
                'name'     : tf.io.FixedLenFeature([], tf.string),
                'cls'      : tf.io.FixedLenFeature([], tf.int64),
                'cls_name' : tf.io.FixedLenFeature([], tf.string),
                'image'    : tf.io.FixedLenFeature([], tf.string),
            }
            parsed_features = tf.io.parse_single_example(x, keys_to_features)
            parsed_features['image'] = tf.image.decode_png(parsed_features['image'], channels=3)
            return (
                parsed_features['name'],
                parsed_features['cls'],
                parsed_features['cls_name'],
                parsed_features['image'],
            )
        if name == 'test':
            batch_size = 1
            count = 1
        else:
            batch_size = self._batch_size
            count = self._count
        dataset = tf.data.TFRecordDataset(full_record_name, num_parallel_reads=self._num_parallel_reads)
        dataset = dataset.map(parse, num_parallel_calls=self._num_parallel_calls)
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(self._shuffle_buffer)
        dataset = dataset.repeat(count=count)
        dataset = dataset.prefetch(buffer_size=self._prefatch_buffer_size)

        for name, cls, cls_name, image in iter(dataset):
            image = tf.reshape(image, [-1, *self._image_size, 3])
            image = tf.cond(tf.random.uniform([], 0, 2, dtype=tf.int32),
                            true_fn=lambda: tf.image.flip_left_right(image),
                            false_fn=lambda: image)

            image = tf.image.rot90(image, k=tf.random.uniform([], 0, 2, dtype=tf.int32))
            image = tf.cast(image, dtype=tf.float32)
            image = image / 255

            yield name, tf.one_hot(cls, depth=len(classes_decode)), cls_name, image
