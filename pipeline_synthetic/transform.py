import logging
from pathlib import Path
from pickle import dump

import mlflow
import numpy as np
from sklearn import preprocessing


def main():

    logger.info("Loading x training and test data")
    x_train = np.load(output_path.joinpath("x_train_chunks.npy"))
    x_test = np.load(output_path.joinpath("x_test_chunks.npy"))

    transformer = preprocessing.PowerTransformer(method="yeo-johnson")

    logger.info("Fitting and transforming x training and then applying to test data")
    for channel_number in range(x_train.shape[-1]):
        logger.info("Transforming channel %s", channel_number)
        transformer = preprocessing.PowerTransformer(method="yeo-johnson")

        x_train[:, :, channel_number] = transformer.fit_transform(
            x_train[:, :, channel_number]
        )

        x_test[:, :, channel_number] = transformer.transform(
            x_test[:, :, channel_number]
        )

        dump(
            transformer,
            open(output_path.joinpath(f"transformer_{channel_number}.pkl"), "wb"),
        )
        logger.info("Transformed channel %s. Logging to MLFlow", channel_number)
        mlflow.log_artifact(output_path.joinpath(f"transformer_{channel_number}.pkl"))

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
        filename=script_dir.joinpath("transform.log"),
        filemode="w",
    )
    logger = logging.getLogger(__name__)

    main()
    mlflow.log_artifact(script_dir.joinpath("transform.log"))
