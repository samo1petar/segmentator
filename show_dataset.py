import os
import cv2
import numpy as np
from IPython import embed


def show(image):
    cv2.imshow('', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    dataset_dir = '/home/petar/Projects/Shapes/presentations_png_mmseg 6'

    for image_name in os.listdir(os.path.join(dataset_dir, 'images')):
        image = cv2.imread(os.path.join(dataset_dir, 'images', image_name))
        label = cv2.imread(os.path.join(dataset_dir, 'labels', image_name))

        show(image)
        show(label)

        mask_rec = np.zeros_like(label)
        mask_rec[np.where((label == [0, 0, 128]).all(axis=2))] = [255, 255, 255]
        mask_rec = mask_rec[:, :, 0]

        mask_oval = np.zeros_like(label)
        mask_oval[np.where((label == [0, 128, 0]).all(axis=2))] = [255, 255, 255]
        mask_oval = mask_oval[:, :, 0]

        show(mask_rec)
        show(mask_oval)

        mask = np.stack([mask_rec, mask_oval], axis=-1)

        print (mask.shape)

    exit()

    image = cv2.imread('/home/petar/Projects/Shapes/presentations_png_mmseg 6/images/pres_9_full_screen_slide_3.png')

    canny_image = cv2.Canny(image, 0, 250)

    show(canny_image)

    label = cv2.imread('/home/petar/Projects/Shapes/presentations_png_mmseg 6/labels/pres_9_full_screen_slide_3.png',
                       cv2.IMREAD_COLOR)
    label = cv2.resize(label, (768, 512))
    mask = np.zeros_like(label)
    mask[np.where((label == [0, 0, 128]).all(axis=2))] = [255, 255, 255]
    mask = mask[:, :, 0]

    canny_mask = cv2.Canny(mask, 0, 250)
    show(canny_mask)

    blur = cv2.GaussianBlur(canny_mask, (15, 15), 0).astype(np.int32) * 5
    blur[blur > 255] = 255
    blur = blur.astype(np.uint8)

    new_mask = np.maximum(mask, blur)

    show(blur)

    show(new_mask)

    # show(label)

    # embed()