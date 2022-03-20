import numpy as np
import numpy.typing as npt
import segyio
from skimage import color, io


def load_section(
    section_number: int, x_path: str, y_path: str, attribute_path: str
) -> tuple:
    """Loads the GPR section, the ground truth, and the attributes.

    Args:
        section_number (int): a number between 89 and 96.
        x_path (str): local path in which the gpr section is stored.
        y_path (str): local path in which the ground truth is stored.
        attribute_path (str): local path in which the attributes are stored.

    Returns:
        tuple: (gpr_section, ground_truth, attributes)
    """

    # Load the GPR section
    x = segy_to_numpy(load_segy(x_path + "FILE0" + str(section_number) + ".SGY"))
    # Load the ground truth
    y = load_ground_truth(y_path + "0" + str(section_number) + ".jpg")
    # Load the attributes
    attributes = []
    for att in ["Sim", "En", "InstFreq", "InstPha", "Hilb"]:
        attributes.append(
            np.genfromtxt(attribute_path + att + str(section_number) + ".dat").T
        )

    return x, y, attributes


def load_ground_truth(image_path: str) -> npt.NDArray:
    """Loads the ground truth image.

    Args:
        image_path (str): local path in which the ground truth is stored.

    Returns:
        npt.NDArray: a numpy array containing the ground truth.
    """

    array = color.rgb2gray(io.imread(image_path))
    array = np.where(array >= 0.5, 1, 0)

    return array


def load_segy(path: str) -> segyio.SegyFile:
    """Loads a segy file.

    Args:
        path (str): path to the segy file.

    Returns:
        segyio.SegyFile: a segy file.
    """

    segy_object = segyio.open(path, ignore_geometry=True)

    return segy_object


def segy_to_numpy(segy_file: segyio.SegyFile) -> npt.NDArray:
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
