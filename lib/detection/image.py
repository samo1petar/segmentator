import cv2
import numpy as np
from typing import Tuple


def is_image(image: np.ndarray) -> bool:
    if image.ndim == 2: return True
    if image.ndim == 3 and image.shape[2] in [1, 3]:
        return True
    return False


def imread(image_path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    return cv2.imread(filename = image_path, flags = flags)


def show(
        image   : np.ndarray,
        title   : str  = '',
        wait    : int  = 0,
        destroy : bool = True,
) -> None:
    assert is_image(image)
    cv2.imshow(title, image)
    cv2.waitKey(wait)
    if destroy:
        cv2.destroyAllWindows()


def show_and_return_key(
        image   : np.ndarray,
        title   : str  = '',
        wait    : int  = 0,
        destroy : bool = True,
) -> int:
    assert is_image(image)
    cv2.imshow(title, image)
    key = cv2.waitKey(wait)
    if destroy:
        cv2.destroyAllWindows()
    return key


def resize(
        image : np.ndarray,
        size  : Tuple[int] = None,
        dx    : float      = 0.0,
) -> np.ndarray:
    assert is_image(image)
    assert size is not None and dx == 0.0 or size is None and dx > 0.0

    if size:
        return cv2.resize(image, size)

    return cv2.resize(image, (int(image.shape[1] * dx), int(image.shape[0] * dx)))


def dewarp(image, points):

    pts1 = np.float32(points.reshape(4, 2))
    pts2 = np.float32([[0, 0], [0, 640], [400, 0], [400, 640]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(image, matrix, (400, 640))

    return result