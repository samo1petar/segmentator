from lib.loader_coco.RecordReader import RecordReader
from lib.loader_coco.RecordWriter import RecordWriter
from gluoncv import utils
from IPython import embed
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
import numpy as np
import json
from lib.loader_coco.classes import classes

writer = RecordWriter(
    data_path           = '/media/david/A/Dataset/COCO',
    record_dir          = 'records',
    record_name         = 'one_image',
    save_n_test_images  = 1,
    save_n_train_images = 1,
)

reader = RecordReader(
    record_dir           = 'records',
    record_name          = 'one_image',
    batch_size           = 1,
    shuffle_buffer       = 1,
    num_parallel_calls   = 1,
    num_parallel_reads   = 1,
    prefatch_buffer_size = 1,
    count                = 1,
)

test_record = reader.read_record('test')

for index, class_ids, bboxes, image in test_record:

    image = (image[0].numpy() * 255).astype(np.uint8)

    bboxes = bboxes.numpy()

    class_ids = class_ids.numpy()

    utils.viz.plot_bbox(image, bboxes, scores=None, labels=class_ids, class_names=list(classes.values()))
    plt.show()