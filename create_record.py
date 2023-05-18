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
        images_dir='/home/petar/Projects/Shapes/presentations_png_mmseg 6/images',
        masks_dir='/home/petar/Projects/Shapes/presentations_png_mmseg 6/labels',
    )

train_set, test_set = split.create_splits()

writer = RecordWriter(
    data_path           = '/home/petar/Projects/Shapes/presentations_png_mmseg 6/',
    record_dir          = '/home/petar/Projects/Shapes/records',
    record_name         = 'segmentator_full_gray_rect_oval_smooth_test',
    train_set           = train_set,
    test_set            = test_set,
    save_n_test_images  = 20,
    save_n_train_images = 20,
    image_size          = (512, 768),
)

reader = RecordReader(
    record_dir           = '/home/petar/Projects/Shapes/records',
    record_name          = 'segmentator_full_gray_rect_oval_smooth_test',
    batch_size           = 1,
    shuffle_buffer       = 1,
    num_parallel_calls   = 1,
    num_parallel_reads   = 1,
    prefatch_buffer_size = 1,
    count                = 1,
    image_size           = (512, 768),
)

test_record = reader.read_record('train')

def show(image):
    cv2.imshow('', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

for name, image, mask in test_record:

    image = (image[0].numpy() * 255).astype(np.uint8)

    show(image)

    mask = (mask[0].numpy() * 255).astype(np.uint8)

    merge = image * 0.5 + mask * 0.5
    merge = merge.astype(np.uint8)

    print (name)

    show(merge)
