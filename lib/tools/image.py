import cv2
import imutils
import numpy as np


def save_image(image: np.ndarray, path: str) -> None:
    cv2.imwrite(path, image)


def save_numpy_image(image: np.ndarray, path: str) -> None:
    np.save(path, image)


def flip(image: np.ndarray) -> np.ndarray:
    return np.flip(image, axis=1)


def rotate_90(image: np.ndarray) -> np.ndarray:
    return imutils.rotate_bound(image, -90)


def resize_keep_aspect(image: np.ndarray, resize: int, size: str = 'bigger') -> np.ndarray:
    assert size in ['bigger', 'smaller']
    height, width, _ = image.shape

    if size == 'bigger':
        if height > width:
            image = cv2.resize(image, (resize, int(image.shape[0] / image.shape[1] * resize)))
        else:
            image = cv2.resize(image, (int(image.shape[1] / image.shape[0] * resize), resize))
    elif size == 'smaller':
        if height > width:
            image = cv2.resize(image, (int(image.shape[1] / image.shape[0] * resize), resize))
        else:
            image = cv2.resize(image, (resize, int(image.shape[0] / image.shape[1] * resize)))
    return image


def pad_to_fit(image: np.ndarray, size: int) -> np.ndarray:

    if image.shape[0] == size and image.shape[1] == size:
        return image

    # Top
    zeros = np.zeros((image.shape[0], (size - image.shape[1]) // 2, image.shape[2])).astype(np.uint8)
    image = np.hstack((zeros, image))
    # Bottom
    zeros = np.zeros((image.shape[0], size - image.shape[1], image.shape[2])).astype(np.uint8)
    image = np.hstack((image, zeros))
    # Left
    zeros = np.zeros(((size - image.shape[0]) // 2, image.shape[1], image.shape[2])).astype(np.uint8)
    image = np.vstack((zeros, image))
    # Right
    zeros = np.zeros((size - image.shape[0], image.shape[1], image.shape[2])).astype(np.uint8)
    image = np.vstack((image, zeros))
    return image
