import logging
from pathlib import Path
from pickle import dump

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig
from sklearn import preprocessing


@hydra.main(config_path=".", config_name="config.yaml")
def main(config: DictConfig):

    x_train_sampled = []

    for number in range(89, 97):

        if number != 89 and number != 96:

            x_data = np.load(output_path / f"x_data_{number}.npy")
            logger.info("Using section %s in the training set", number)
            x_train_sampled.append(x_data)

    stacked_x_train = np.vstack(
        [data.reshape(-1, data.shape[-1]) for data in x_train_sampled]
    )
    logger.info("Fitting and transforming x training")
    transformer = preprocessing.PowerTransformer(method="yeo-johnson")
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    logger.info("Using MinMaxScaler with feature range -1, 1")
    transformer = transformer.fit(stacked_x_train)
    transformed_x_train = transformer.transform(stacked_x_train)
    scaler = scaler.fit(transformed_x_train)

    dump(
        transformer,
        open(output_path.joinpath(f"transformer.pkl"), "wb"),
    )
    logger.info("Logging transformer to MLflow")
    mlflow.log_artifact(output_path.joinpath(f"transformer.pkl"))
    dump(scaler, open(output_path.joinpath(f"scaler.pkl"), "wb"))
    logger.info("Logging scaler to MLflow")
    mlflow.log_artifact(output_path.joinpath(f"scaler.pkl"))

    logger.info("Applying transformation to training data and test data")
    for number in range(89, 97):
        logger.info(f"Transforming data %s", number)
        data = np.load(output_path / f"x_data_{number}.npy")
        original_shape = data.shape
        transformed_data = transformer.transform(data.reshape(-1, original_shape[-1]))
        transformed_data = scaler.transform(transformed_data)
        transformed_data = transformed_data.reshape(original_shape)

        logger.info(f"Saving transformed data to disk")
        np.save(output_path / f"x_data_transformed_{number}.npy", transformed_data)


if __name__ == "__main__":

    script_dir = Path(__file__).parent.absolute()
    output_path = script_dir.joinpath("processed_data/")
    root_path = script_dir.parent.absolute()

    mlflow.set_tracking_uri(f"{root_path}/mlruns")

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
    mlflow.log_artifact("transform.log")
