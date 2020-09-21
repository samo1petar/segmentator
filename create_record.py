from lib.loader.RecordReader import RecordReader
from lib.loader.RecordWriter import RecordWriter
from IPython import embed
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
import numpy as np
import json
from lib.split.split import Split


split = Split(
        images_dir='/media/david/A/Datasets/PlayHippo/images',
        masks_dir='/media/david/A/Datasets/PlayHippo/masked_images',
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
    import cv2
    cv2.imshow('', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def fillhole(input_image):
    '''
    input gray binary image  get the filled image by floodfill method
    Note: only holes surrounded in the connected regions will be filled.
    :param input_image:
    :return:
    '''
    im_flood_fill = input_image.copy()
    h, w = input_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv
    return img_out


for name, image, mask in test_record:
    import cv2
    image = (image[0].numpy() * 255).astype(np.uint8)
    mask = (mask[0].numpy() * 255).astype(np.uint8)

    print (name)

    # mask_of_mask = np.ones_like(mask)
    # mask_of_mask *= 255
    # mask_of_mask[mask > 10] = 0

    show(mask)

    m = mask.copy()
    m = m.astype(np.bool)
    m = m.astype(np.uint8) * 255

    show(m)

    show(fillhole(m[:, :, 0]))
