import numpy as np
import numpy.typing as npt


def clip(array: npt.NDArray, sample_size: int) -> npt.NDArray:
    """Clips the array to a size divisible by the sample size.

    Args:
        array (npt.NDArray): a numpy array containing the data
        (attribute, gpr, or ground truth).
        sample_size (int): the size of the sample, typically 16 or 32.

    Returns:
        npt.NDArray: the same array, clipped to a size divisible by the sample
    """
    array = array[
        : ((array.shape[-2] // sample_size) * sample_size),
        : ((array.shape[-1] // sample_size) * sample_size),
    ]

    return array


def write_metadata(
    output_path,
    sample_size,
    sample_density,
    min_max_scale,
    loc,
    sigma,
    noise,
    x_train,
    x_val,
    x_test,
):
    with open(output_path + "dataset_description.txt", "w") as f:
        f.write("Dataset description:\n")
        f.write(
            "This dataset contains the following attributes in order inside the array:\n"
        )
        # Modificar aqui caso sejam adicionados mais atributos
        f.write("Similarity, Energy, Instantaneous Frequency,\n")
        f.write("Instantaneous Phase, Hilbert/Similarity\n")
        f.write(f"Number of examples in the train dataset {x_train.shape[0]}\n")
        f.write(f"Number of examples in the validation dataset {x_val.shape[0]}\n")
        f.write(f"Number of examples in the test dataset {x_test.shape[0]}\n")
        # Modificar aqui caso altere a seção de teste
        f.write("Test section was section number 089 and 096\n")
        f.write(f"Sample size is {sample_size}x{sample_size}\n")
        f.write(f"Sample density is {sample_density}\n")
        if noise:
            f.write("Noise was used\n")
            f.write(f"Noise parameters used were: {loc} and {sigma}\n")
        else:
            f.write("No noise was added\n")
        if min_max_scale:
            f.write("Scaled between 0 and 1\n")


def save_processed_data(
    output_path,
    loaded_sections,
    data_normalized,
    x_train,
    x_val,
    y_train,
    y_val,
    x_test,
    y_test,
):
    np.save(output_path + "x_train.npy", x_train)
    np.save(output_path + "y_train.npy", y_train)
    np.save(output_path + "x_val.npy", x_val)
    np.save(output_path + "y_val.npy", y_val)
    np.save(output_path + "x_test.npy", x_test)
    np.save(output_path + "y_test.npy", y_test)

    np.save(output_path + "x_test_section_89.npy", data_normalized[0])
    np.save(output_path + "y_test_section_89.npy", loaded_sections[89]["y"])
    np.save(output_path + "x_test_section_96.npy", data_normalized[-1])
    np.save(output_path + "y_test_section_96.npy", loaded_sections[96]["y"])
