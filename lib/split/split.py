import os
from random import shuffle


class Split:
    def __init__(
            self,
            images_dir : str,
            masks_dir  : str,
    ):
        self._images_dir = images_dir
        self._masks_dir = masks_dir


    def create_data(self):
        all_images = {}

        for root, subdir, files in os.walk(self._images_dir):
            for file in files:
                file_path = os.path.join(root, file)
                key = '/'.join(file_path.rsplit('/', 3)[1:])
                all_images[key] = file_path

        all_masks = {}
        for root, subdir, files in os.walk(self._masks_dir):
            for file in files:
                file_path = os.path.join(root, file)
                key = '/'.join(file_path.rsplit('/', 3)[1:])
                all_masks[key] = file_path

        self.data = {key: {'image': all_images[key], 'mask': all_masks[key]} for key in all_masks if key in all_images}


    def create_splits(self, train_perc : float = 0.8):

        self.create_data()

        keys = list(self.data.keys())
        shuffle(keys)

        train_keys = keys[:int(train_perc * len(keys)) ]
        test_keys  = keys[ int(train_perc * len(keys)):]

        train_set = {}
        for key in train_keys:
            train_set[key] = self.data[key]

        test_set = {}
        for key in test_keys:
            test_set[key] = self.data[key]

        return train_set, test_set


    def save(self, path): # TODO
        ...
        # with open(path, 'w') as f:
        #     f.writelines('\n'.join(self._train_data))


if __name__ == '__main__':

    split = Split(
        images_dir='/media/david/A/Datasets/PlayHippo/images',
        masks_dir='/media/david/A/Datasets/PlayHippo/masked_images',
    )

    train_set, test_set = split.create_splits()

    from IPython import embed
    embed()