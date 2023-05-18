import cv2
import numpy as np
from IPython import embed


def show(image):
    cv2.imshow('', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    image = cv2.imread('/home/petar/Projects/Shapes/presentations_png_mmseg 6/images/pres_9_full_screen_slide_3.png',
                       cv2.IMREAD_GRAYSCALE)

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