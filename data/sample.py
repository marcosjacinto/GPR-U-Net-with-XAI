import numpy as np

from utils import clip


def sample_input_data(
    array: np.ndarray, sample_size: int, step_div: int = 1
) -> np.ndarray:

    total = 0
    step = int(sample_size / step_div)
    shape1, shape2, shape3 = array.shape[0], array.shape[1], array.shape[2]

    for i in range(int(shape1 / (step)) - (step_div - 1)):
        for w in range(int(shape2 / (step)) - (step_div - 1)):
            total += 1

    x_sampled = np.empty((total, sample_size, sample_size, shape3))

    n = 0
    for i in range(int(shape1 / (step)) - (step_div - 1)):
        istart = int(step * i)
        iend = int(istart + sample_size)
        for w in range(int(shape2 / (step)) - (step_div - 1)):
            wstart = int(step * w)
            wend = int(wstart + sample_size)
            x_sampled[n, ...] = array[istart:iend, wstart:wend].reshape(
                1, sample_size, sample_size, shape3
            )
            n += 1

    return x_sampled


def sample_ground_truth(
    array: np.ndarray, sample_size: int, step_div: int = 1
) -> np.ndarray:

    array = clip(array, 16)

    total = 0
    step = int(sample_size / step_div)
    shape1, shape2 = array.shape[0], array.shape[1]

    for i in range(int(shape1 / (step)) - (step_div - 1)):
        for w in range(int(shape2 / (step)) - (step_div - 1)):
            total += 1

    y_sampled = np.empty((total, sample_size, sample_size))

    n = 0
    for i in range(int(shape1 / (step)) - (step_div - 1)):
        istart = int(step * i)
        iend = int(istart + sample_size)
        for w in range(int(shape2 / (step)) - (step_div - 1)):
            wstart = int(step * w)
            wend = int(wstart + sample_size)
            y_sampled[n, ...] = array[istart:iend, wstart:wend].reshape(
                1, sample_size, sample_size
            )
            n += 1

    return y_sampled
