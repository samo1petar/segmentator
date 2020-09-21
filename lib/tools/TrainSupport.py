import cv2
from os import listdir, mkdir, remove
from os.path import join
from shutil import copyfile
import tensorflow as tf
from typing import Iterator
from lib.tools.file import mkdir, choose_one_from_dir
from lib.tools.time import get_time
from lib.tools.softmax import softmax
from lib.tools.plot import save_figure


class TrainSupport:
    def __init__(
            self,
            save_dir : str = 'results',
            name     : str = '',
    ):
        self._save_dir = join(save_dir, get_time() + '_' + name)
        self.model_saving_dir = join(self._save_dir, 'model')
        self.sample_train_dir = join(self._save_dir, 'sample_train')
        self.sample_test_dir  = join(self._save_dir, 'sample_test')

        for path in [self._save_dir, self.model_saving_dir, self.sample_train_dir, self.sample_test_dir]:
            mkdir(path)
        copyfile('params.py', join(self._save_dir, 'params.py'))

    def restore(self, save_dir):
        experiment = choose_one_from_dir(save_dir)

        checkpoint_files = [x for x in listdir(join(experiment, 'model'))]

        for x in checkpoint_files:
            copyfile(join(experiment, 'model', x), join(self.model_saving_dir, x))
        for x in listdir(join(experiment, 'metrics')):
            copyfile(join(experiment, 'metrics', x), join(self._save_dir, 'metrics', x))

        return join(experiment, 'model')

    @staticmethod
    def sample_from(model: tf.keras.Model, iterator : Iterator, save_dir: str, save_count: int = 20):
        for x in listdir(save_dir):
            remove(join(save_dir, x))

        count = 0
        for name, image, mask in iterator:

            prediction = model(image)

            prediction = prediction.numpy()
            name = name.numpy()
            image = image.numpy()

            name_ = name.decode('utf8')

            image_ = image

            from IPython import embed
            embed()
            exit()
