import numpy as np


def softmax(array: np.ndarray) -> np.ndarray:
    return np.power(np.e, array) / np.sum(np.power(np.e, array))
