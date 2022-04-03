import logging
from pathlib import Path

import mlflow
import numpy as np

from gpr_unet import data_augmentation


def main():

    logger.info("Loading transformed data and ground truth")
    x_train = np.load(output_path.joinpath("x_train_sampled.npy"))
    y_train = np.load(output_path.joinpath("y_train_sampled.npy"))

    logger.info("Augmenting data")
    mean = 0.0
    standard_deviation = 0.1
    apply_noise = True
    if apply_noise:
        logger.info(
            "Inserting noise in training data and using mean: %s and standard deviation: %s",
            mean,
            standard_deviation,
        )

    x_train_augmented = data_augmentation.augment_data(
        x_train,
        mean=mean,
        standard_deviation=standard_deviation,
        apply_noise=apply_noise,
    )
    y_train_augmented = data_augmentation.augment_data(y_train, apply_noise=False)

    logger.info("Saving augmented data to disk")
    np.save(output_path.joinpath("x_train_augmented.npy"), x_train_augmented)
    np.save(output_path.joinpath("y_train_augmented.npy"), y_train_augmented)


if __name__ == "__main__":

    script_dir = Path(__file__).parent.absolute()
    output_path = script_dir.joinpath("processed_data/")

    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        filename=script_dir.joinpath("augment.log"),
        filemode="w",
    )
    logger = logging.getLogger(__name__)

    main()
    mlflow.log_artifact(script_dir.joinpath("augment.log"))
