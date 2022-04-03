from pathlib import Path

import numpy as np
import segyio
from skimage import color, io


def load_segy(path: str) -> segyio.SegyFile:
    """Loads a segy file.

    Args:
        path (str): path to the segy file.

    Returns:
        segyio.SegyFile: a segy file.
    """
    segy_object = segyio.open(path, ignore_geometry=True)

    return segy_object


def segy_to_numpy(segy_file: segyio.SegyFile) -> np.ndarray:
    """Converts a segy file to a numpy array.

    Args:
        segy_file (segyio.SegyFile): a SegyFile object.

    Returns:
        npt.NDArray: a numpy array converted from the segy file.
    """
    n_samples: int = len(segy_file.samples)  # type: ignore
    n_traces: int = len(segy_file.trace)
    shape = (n_samples, n_traces)
    array = np.empty(shape)

    for n in range(0, len(segy_file.trace)):
        array[:, n] = segy_file.trace[n]

    return array


def gpr_data_to_numpy(file_path: str) -> np.ndarray:
    """Load the GPR section or an attribute.

    Args:
        file_path (str): path to *.sgy, *.SGY, or *.dat file.

    Returns:
        np.ndarray: data as a numpy array.
    """
    if file_path.endswith(".SGY") or file_path.endswith(".sgy"):
        segy_file = segyio.open(file_path, ignore_geometry=True)
        array = segy_to_numpy(segy_file)
        return array
    elif file_path.endswith(".dat"):
        array = np.genfromtxt(file_path)
        return array


def load_ground_truth(image_path: str) -> np.ndarray:
    """Loads the ground truth image.

    Args:
        image_path (str): local path in which the ground truth is stored.

    Returns:
        np.ndarray: a numpy array containing the ground truth.
    """
    array = color.rgb2gray(io.imread(image_path))
    array = np.where(array >= 0.5, 1, 0)

    return array


def pad_attribute(attribute: np.ndarray, number_of_rows: int) -> np.ndarray:
    """Pad the attribute to a size divisible by the number of rows.

    Args:
        attribute (np.ndarray): the attribute to be padded.
        number_of_rows (int): the number of rows of reference, usually the number of
        rows in the GPR section.

    Returns:
        np.ndarray: the same attribute, but padded to a size divisible by the number
        of rows.
    """
    difference = number_of_rows - attribute.shape[0]
    # divide by two because we will need to pad both sides
    missing_rows = int(difference / 2)

    if missing_rows > 0:
        attribute = np.pad(
            attribute, ((missing_rows, missing_rows), (0, 0)), constant_values=0
        )

    return attribute
