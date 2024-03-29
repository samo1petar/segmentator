import os
import tensorflow as tf
from typing import Generator, Tuple


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

    def read_record(self, set_name: str) -> Generator[tf.Tensor, None, None]:

        assert set_name in ['train', 'test']

        full_record_name = os.path.join(self._record_dir, self._record_name + '_' + set_name + '.tfrecord')

        def parse(x):
            keys_to_features = {
                'name'     : tf.io.FixedLenFeature([], tf.string),
                'image'    : tf.io.FixedLenFeature([], tf.string),
                'mask'    : tf.io.FixedLenFeature([], tf.string),
            }
            parsed_features = tf.io.parse_single_example(x, keys_to_features)
            parsed_features['image'] = tf.image.decode_png(parsed_features['image'], channels=3)
            parsed_features['mask'] = tf.image.decode_png(parsed_features['mask'], channels=3)
            return (
                parsed_features['name'],
                parsed_features['image'],
                parsed_features['mask'],
            )
        if set_name == 'test':
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

        for name, image, mask in iter(dataset):

            # flip_cond = tf.random.uniform([], 0, 2, dtype=tf.int32)
            # rot_cond = tf.random.uniform([], 0, 4, dtype=tf.int32)

            # if set_name == 'train':
            #
            #     image_2 = tf.image.random_brightness(image, 0.2)
            #     image_2 = tf.image.random_hue(image_2, 0.2)
            #     image_2 = tf.image.random_contrast(image_2, 0.0, 0.3)
            #     image_2 = tf.image.random_saturation(image_2, 2, 10)
            #
            #     image = tf.cond(
            #         tf.random.uniform([], 0, 2, dtype=tf.int32),
            #         true_fn=lambda: image,
            #         false_fn=lambda: image_2,
            #     )

            # image = tf.reshape(image, [-1, *self._image_size, 3])
            image = tf.image.resize(image, [*self._image_size])
            # image = tf.cond(flip_cond,
            #                 true_fn=lambda: tf.image.flip_left_right(image),
            #                 false_fn=lambda: image)

            # image = tf.image.rot90(image, k=rot_cond)
            image = tf.cast(image, dtype=tf.float32)
            image = image / 255

            # mask = tf.reshape(mask, [-1, *self._image_size, 1])
            mask = tf.image.resize(mask, [*self._image_size])

            mask = mask[..., 1:3]
            # mask = tf.cond(flip_cond,
            #                 true_fn=lambda: tf.image.flip_left_right(mask),
            #                 false_fn=lambda: mask)
            # mask = tf.image.rot90(mask, k=rot_cond)
            mask = tf.cast(mask, dtype=tf.float32)
            mask = mask / 255

            yield name, image, mask
