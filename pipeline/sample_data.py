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

    sample_size = config["sampling"]["sample_size"]
    sampling_step = config["sampling"]["sampling_step"]

    x_train_list, y_train_list = [], []
    x_test_list, y_test_list = [], []

    for number in range(89, 97):

        x_data = np.load(output_path.joinpath(f"x_data_transformed_{number}.npy"))
        y_data = np.load(output_path.joinpath(f"y_data_{number}.npy"))
        y_data = np.expand_dims(y_data, axis=-1)

        x_data_sampled = sample.sample_data(x_data, sample_size, sampling_step)
        y_data_sampled = sample.sample_data(y_data, sample_size, sampling_step)

        if number != 89 and number != 96:
            x_train_list.append(x_data_sampled)
            y_train_list.append(y_data_sampled)
        else:
            x_test_list.append(x_data_sampled)
            y_test_list.append(y_data_sampled)

    logger.info("Creating training and test data")
    x_train = np.vstack(x_train_list)
    y_train = np.vstack(y_train_list)
    x_test = np.vstack(x_test_list)
    y_test = np.vstack(y_test_list)

    logger.info("Creating validation set from training set")
    x_train, x_val, y_train, y_val = model_selection.train_test_split(
        x_train, y_train, test_size=0.15, random_state=42
    )

    logger.info("Saving sampled data to disk")
    np.save(output_path.joinpath("x_train_sampled.npy"), x_train)
    np.save(output_path.joinpath("x_val_sampled.npy"), x_val)
    np.save(output_path.joinpath("x_test_sampled.npy"), x_test)
    np.save(output_path.joinpath("y_train_sampled.npy"), y_train)
    np.save(output_path.joinpath("y_val_sampled.npy"), y_val)
    np.save(output_path.joinpath("y_test_sampled.npy"), y_test)

    logger.info("X_train shape: %s", x_train.shape)
    logger.info("X_val shape: %s", x_val.shape)
    logger.info("X_test shape: %s", x_test.shape)
    logger.info("Y_train shape: %s", y_train.shape)
    logger.info("Y_val shape: %s", y_val.shape)
    logger.info("Y_test shape: %s", y_test.shape)


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
