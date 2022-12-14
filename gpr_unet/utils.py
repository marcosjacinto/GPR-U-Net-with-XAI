import numpy as np


def clip(array: np.ndarray, sample_size: int) -> np.ndarray:
    """Clips the array to a size divisible by the sample size.

    Args:
        array (np.ndarray): a numpy array containing the data
        (attribute, gpr, or ground truth).
        sample_size (int): the size of the sample, typically 16 or 32.

    Returns:
        np.ndarray: the same array, clipped to a size divisible by the sample
    """
    array = array[
        : ((array.shape[-2] // sample_size) * sample_size),
        : ((array.shape[-1] // sample_size) * sample_size),
    ]

    return array
