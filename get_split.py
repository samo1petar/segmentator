from lib.loader.RecordReader import RecordReader
from IPython import embed
import os
import cv2
import numpy as np
import json



reader = RecordReader(
    record_dir='/home/petar/Projects/Shapes/records',
    record_name='segmentator_full_gray_rect_oval_smooth',
    batch_size=1,
    shuffle_buffer=1,
    num_parallel_calls=1,
    num_parallel_reads=1,
    prefatch_buffer_size=1,
    count=1,
    image_size=(512, 768),
)


record = reader.read_record('test')

def show(image):
    cv2.imshow('', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

split = dict()

for name, _, _ in record:

    name_str = name.numpy()[0].decode('utf-8')

    split[name_str] = {
        'image': os.path.join("/home/petar/Projects/Shapes/presentations_png_mmseg 6/images/", name_str),
        'mask': os.path.join("/home/petar/Projects/Shapes/presentations_png_mmseg 6/labels/", name_str),
    }


with open('../segmentator_pytorch/test_split.json', 'w') as f:
    json.dump(split, f, indent=4)
