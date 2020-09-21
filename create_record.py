from lib.loader.RecordReader import RecordReader
from lib.loader.RecordWriter import RecordWriter
from IPython import embed
import matplotlib
matplotlib.use('tkagg')
import cv2
from matplotlib import pyplot as plt
import numpy as np
import json
from lib.split.split import Split


split = Split(
        images_dir='/media/david/A/Datasets/PlayHippo/images',
        masks_dir='/media/david/A/Datasets/PlayHippo/masked_images_cleaned',
    )

train_set, test_set = split.create_splits()

writer = RecordWriter(
    data_path           = '/media/david/A/Dataset/PlayHippo',
    record_dir          = 'records',
    record_name         = 'data',
    train_set           = train_set,
    test_set            = test_set,
    save_n_test_images  = 1,
    save_n_train_images = 1,
)

reader = RecordReader(
    record_dir           = 'records',
    record_name          = 'data',
    batch_size           = 1,
    shuffle_buffer       = 1,
    num_parallel_calls   = 1,
    num_parallel_reads   = 1,
    prefatch_buffer_size = 1,
    count                = 1,
)

test_record = reader.read_record('train')

def show(image):
    cv2.imshow('', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

for name, image, mask in test_record:
    image = (image[0].numpy() * 255).astype(np.uint8)
    mask = (mask[0].numpy() * 255).astype(np.uint8)

    merge = image * 0.5 + mask * 0.5
    merge = merge.astype(np.uint8)

    print (name)

    show(merge)
