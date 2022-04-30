import logging
from pathlib import Path

import hydra
import mlflow
import numpy as np
from gpr_unet import sample
from omegaconf import DictConfig
from sklearn import model_selection


@hydra.main(config_name="config")
def main(config: DictConfig):

    logger.info("Loading transformed data and ground truth")
    x_train = np.load(output_path.joinpath("x_train_chunks_transformed.npy"))
    y_train = np.load(output_path.joinpath("y_train_chunks.npy"))
    x_test = np.load(output_path.joinpath("x_test_chunks_transformed.npy"))
    y_test = np.load(output_path.joinpath("y_test_chunks.npy"))

    logger.info("Sampling data")
    sample_size = config["sampling"]["sample_size"]
    sampling_step = config["sampling"]["sampling_step"]
    x_train = sample.sample_data(x_train, sample_size, sampling_step)
    x_test = sample.sample_data(x_test, sample_size, sampling_step)
    y_train = sample.sample_data(y_train, sample_size, sampling_step)
    y_test = sample.sample_data(y_test, sample_size, sampling_step)

    logger.info("Creating validation set from training set")
    x_train, x_val, y_train, y_val = model_selection.train_test_split(
        x_train,
        y_train,
        test_size=0.15,
        random_state=config["sampling"]["random_state"],
    )

    logger.info("Saving sampled data to disk")
    np.save(output_path.joinpath("x_train_sampled.npy"), x_train)
    np.save(output_path.joinpath("x_val_sampled.npy"), x_val)
    np.save(output_path.joinpath("x_test_sampled.npy"), x_test)
    np.save(output_path.joinpath("y_train_sampled.npy"), y_train)
    np.save(output_path.joinpath("y_val_sampled.npy"), y_val)
    np.save(output_path.joinpath("y_test_sampled.npy"), y_test)


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
            logging.FileHandler(filename="sample_data.log", mode="w"),
        ],
    )
    logger = logging.getLogger(__name__)

    main()
    mlflow.log_artifact(script_dir.joinpath("outputs/sample_data.log"))
