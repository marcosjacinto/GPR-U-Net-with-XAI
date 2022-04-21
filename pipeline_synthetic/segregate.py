import logging
import typing as t
from pathlib import Path

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig


@hydra.main(config_name="config")
def main(config: DictConfig):

    np.random.seed(config["segregate"]["random_seed"])

    if config["sampling"]["sample_size"] == 16:
        number_of_samples = 17
    elif config["sampling"]["sample_size"] == 32:
        number_of_samples = 16
    else:
        number_of_samples = config["segregate"]["number_of_samples"]
    logger.info("Dividing data into %s-sample chunks", number_of_samples)
    train_size = int(0.8 * number_of_samples)
    logger.info("Using training size: %s chunks", train_size)

    train_samples = np.random.choice(
        range(number_of_samples), train_size, replace=False
    )
    train_samples = set(train_samples)
    test_samples = set(range(number_of_samples)) - train_samples
    logger.info("Randomly selected train samples: %s", train_samples)
    logger.info("Randomly selected test samples: %s", test_samples)

    x_data = np.load(output_path.joinpath("x_data.npy"))
    y_data = np.load(output_path.joinpath("y_data.npy"))
    y_data = np.expand_dims(y_data, axis=-1)
    logger.info("Loaded data from disk")
    logger.info("X data shape: %s", x_data.shape)
    logger.info("Y data shape: %s", y_data.shape)

    logger.info("Splitting data into training and test sets")
    x_train_chunks, x_test_chunks = train_test_split(
        x_data, train_samples, test_samples, number_of_samples
    )
    y_train_chunks, y_test_chunks = train_test_split(
        y_data, train_samples, test_samples, number_of_samples
    )
    logger.info("Data split into training and test sets. Saving to disk")
    assert (
        x_train_chunks.shape[0] == y_train_chunks.shape[0]
        and x_test_chunks.shape[0] == y_test_chunks.shape[0]
    ), print(
        x_train_chunks.shape[0],
        y_train_chunks.shape[0],
        x_test_chunks.shape[0],
        y_test_chunks.shape[0],
    )
    assert (
        x_train_chunks.shape[1] == y_train_chunks.shape[1]
        and x_test_chunks.shape[1] == y_test_chunks.shape[1]
    ), print(
        x_train_chunks.shape[1],
        y_train_chunks.shape[1],
        x_test_chunks.shape[1],
        y_test_chunks.shape[1],
    )

    np.save(output_path.joinpath("x_train_chunks.npy"), x_train_chunks)
    np.save(output_path.joinpath("x_test_chunks.npy"), x_test_chunks)
    np.save(output_path.joinpath("y_train_chunks.npy"), y_train_chunks)
    np.save(output_path.joinpath("y_test_chunks.npy"), y_test_chunks)


def train_test_split(
    data: np.ndarray,
    train_samples: t.Set[int],
    test_samples: t.Set[int],
    number_of_samples: int = 17,
) -> t.Tuple[np.ndarray, np.ndarray]:

    number_of_traces = data.shape[1]
    step = int(number_of_traces / number_of_samples)
    data_chunks = {}
    start = 0
    for sample_number in range(number_of_samples):
        data_chunks[sample_number] = data[:, start : start + step, :]
        start += step

    train_data = np.hstack(
        [data_chunks[sample_number] for sample_number in train_samples]
    )
    test_data = np.hstack(
        [data_chunks[sample_number] for sample_number in test_samples]
    )

    return train_data, test_data


if __name__ == "__main__":

    script_dir = Path(__file__).parent.absolute()
    output_path = script_dir.joinpath("processed_data/")

    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(filename="segregate.log", mode="w"),
        ],
    )
    logger = logging.getLogger(__name__)

    main()
    mlflow.log_artifact(script_dir.joinpath("outputs/segregate.log"))
