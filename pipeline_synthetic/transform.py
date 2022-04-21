import logging
from pathlib import Path
from pickle import dump

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig
from sklearn import preprocessing


@hydra.main(config_name="config")
def main(config: DictConfig):

    logger.info("Loading x training and test data")
    x_train = np.load(output_path.joinpath("x_train_chunks.npy"))
    x_test = np.load(output_path.joinpath("x_test_chunks.npy"))

    x_train_original_shape = x_train.shape
    x_test_original_shape = x_test.shape

    logger.info("Fitting and transforming x training and then applying to test data")
    transformer = preprocessing.PowerTransformer(method="yeo-johnson")
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    logger.info("Using MinMaxScaler with feature range -1, 1")

    reshaped_x_train = x_train.reshape(-1, x_train_original_shape[-1])
    x_train = transformer.fit_transform(reshaped_x_train)
    x_train = scaler.fit_transform(x_train)
    x_train = x_train.reshape(x_train_original_shape)

    reshaped_x_test = x_test.reshape(-1, x_test_original_shape[-1])
    x_test = transformer.transform(reshaped_x_test)
    x_test = scaler.transform(x_test)
    x_test = x_test.reshape(x_test_original_shape)

    dump(
        transformer,
        open(output_path.joinpath(f"transformer.pkl"), "wb"),
    )
    logger.info("Logging transformer to MLflow")
    mlflow.log_artifact(output_path.joinpath(f"transformer.pkl"))
    dump(scaler, open(output_path.joinpath(f"scaler.pkl"), "wb"))
    logger.info("Logging scaler to MLflow")
    mlflow.log_artifact(output_path.joinpath(f"scaler.pkl"))

    logger.info("Saving transformed data to disk")
    np.save(output_path.joinpath("x_train_chunks_transformed.npy"), x_train)
    np.save(output_path.joinpath("x_test_chunks_transformed.npy"), x_test)


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
            logging.FileHandler(filename="transform.log", mode="w"),
        ],
    )
    logger = logging.getLogger(__name__)

    main()
    mlflow.log_artifact(script_dir.joinpath("outputs/transform.log"))
