import os
import numpy as np
from random import shuffle as shuffle_function
from typing import Dict, Generator, List, Tuple
from lib.loader.info import encode_phone_name


class DataLoader:
    def __init__(
            self,
            data_path: str,
    ):
        self.data_path = data_path
        self.data = {}

        self._sort_functions = {
            encode_phone_name('high_range') : lambda x: int(''.join(x.split('.', 1)[0].split('_')[1:3])),
            encode_phone_name('mid_range')  : lambda x: int(''.join(x.split('.', 1)[0].split('_')[1:3])),
            encode_phone_name('low_range')  : lambda x: int(''.join(x.split('.', 1)[0].split('_')).split('(', 1)[0])
        }

        self.load_into_dict()


    def load_into_dict(self):
        for data_path_x in os.listdir(self.data_path):
            self.data[data_path_x] = {}
            for cls in os.listdir(os.path.join(self.data_path, data_path_x)):
                self.data[data_path_x][cls] = sorted(os.listdir(os.path.join(self.data_path, data_path_x, cls)),
                                                     key=self._sort_functions[data_path_x])


    def get_data_as_list(self, shuffle: bool = False) -> List[str]:
        if len(self.data) == 0:
            return []

        image_path_list = []
        for input_source in self.data.keys():
            for cls in self.data[input_source]:
                image_path_list.extend([os.path.join(self.data_path, input_source, cls, im_name) for im_name in os.listdir(os.path.join(self.data_path, input_source, cls))])

        if shuffle:
            shuffle_function(image_path_list)

        return image_path_list


    def get_annotations_from_path(self, paths: List[str]) -> Dict[str, str]:
        annotations = {}
        for path in paths:
            input_device, cls, im_name = path.rsplit('/', 3)[1:]
            annotations[path] = {
                'cls': cls,
                'im_name': im_name,
                'input_device': input_device,
            }
        return annotations


    def yield_annotations_from_path(self, paths: List[str]) -> Generator[Dict[str, str], None, None]:
        for path in paths:
            input_device, cls, im_name = path.rsplit('/', 3)[1:]
            yield {
                'path': path,
                'cls': cls,
                'im_name': im_name,
                'input_device': input_device,
            }


    def get_next(self, shuffle: bool = False) -> Generator[Tuple[np.ndarray, str, str], None, None]:
        ...


    def get_all(self) -> List[Tuple[np.ndarray, str, str]]:
        ...
