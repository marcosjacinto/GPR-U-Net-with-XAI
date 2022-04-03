import typing as t

import numpy as np


def flip(array: np.ndarray) -> t.Tuple[np.ndarray, np.ndarray]:

    array_horizontally_flipped = np.flip(array, axis=0)
    array_vertically_flipped = np.flip(array, axis=1)

    return array_horizontally_flipped, array_vertically_flipped


def rotate(array: np.ndarray) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:

    array_rotated_90 = np.rot90(array, k=1, axes=(1, 2))
    array_rotated_180 = np.rot90(array, k=2, axes=(1, 2))
    array_rotated_270 = np.rot90(array, k=3, axes=(1, 2))

    return array_rotated_90, array_rotated_180, array_rotated_270


def insert_noise(
    array: np.ndarray,
    mean: float,
    standard_deviation: float,
) -> np.ndarray:

    array_with_noise = array + np.random.normal(
        mean, standard_deviation, size=(array.shape)
    )
    array_with_noise = np.where(array_with_noise < 0, 0, array_with_noise)
    array_with_noise = np.where(array_with_noise > 1, 1, array_with_noise)

    return array_with_noise


def augment_data(
    data: np.ndarray,
    mean: float = 0.0,
    standard_deviation: float = 0.1,
    apply_noise: bool = True,
) -> np.ndarray:

    horizontally_flipped, vertically_flipped = flip(data)
    rotated_90, rotated_180, rotated_270 = rotate(data)

    if apply_noise:
        horizontally_flipped = insert_noise(
            horizontally_flipped, mean, standard_deviation
        )
        vertically_flipped = insert_noise(vertically_flipped, mean, standard_deviation)
        rotated_90 = insert_noise(rotated_90, mean, standard_deviation)
        rotated_180 = insert_noise(rotated_180, mean, standard_deviation)
        rotated_270 = insert_noise(rotated_270, mean, standard_deviation)

    return np.vstack(
        [
            data,
            horizontally_flipped,
            vertically_flipped,
            rotated_90,
            rotated_180,
            rotated_270,
        ]
    )
