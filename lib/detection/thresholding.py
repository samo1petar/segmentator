from lib.detection.image import show
import numpy as np


def threshold_images(image: np.ndarray) -> None:

    blue = image[:, :, 0].copy()
    green = image[:, :, 1].copy()
    red = image[:, :, 2].copy()

    # show(blue, title='blue')
    # show(green, title='green')
    # show(red, title='red')

    blue_mask = blue > 100
    # green_mask = green[green ]
    red_mask = red < 120

    blue[np.invert(blue_mask)] = 0
    show(blue, title='blue masked')

    red[np.invert(red_mask)] = 0
    show(red, title='red masked')

    mask = blue_mask & red_mask

    image_copy = image.copy()

    image_copy[np.invert(mask)] = 0
    show(image_copy)


def threshold_images_2(image: np.ndarray) -> np.ndarray:

    blue = image[:, :, 0].copy().astype(np.int32)
    green = image[:, :, 1].copy().astype(np.int32)
    red = image[:, :, 2].copy().astype(np.int32)

    mask = np.bitwise_and(blue - green > 10, green - red > 20)

    image_copy = image.copy()

    image_copy[np.invert(mask)] = 0
    return image_copy
    # show(image_copy)


def threshold_images_3(image: np.ndarray) -> None:

    blue = image[:, :, 0].copy().astype(np.int32)
    green = image[:, :, 1].copy().astype(np.int32)
    red = image[:, :, 2].copy().astype(np.int32)

    mask = np.bitwise_and(np.abs(red - green) < 30, np.bitwise_and(blue / green < 0.85, green > blue))

    image_copy = image.copy()

    image_copy[np.invert(mask)] = 0
    show(image_copy)
