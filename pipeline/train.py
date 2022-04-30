import logging
import shutil
import os
from pathlib import Path
from pickle import dump

import hydra
import mlflow
import numpy as np
import tensorflow as tf
from gpr_unet.model import build_model
from omegaconf import DictConfig
from tensorflow.keras import Model
from tensorflow.keras.layers import Input


@hydra.main(config_name="config")
def main(config: DictConfig):

    for channel_number in range(0, 7):

        if channel_number != 6:
            run_name = f"Attribute {channel_number}"
        else:
            run_name = "Complete model"
        with mlflow.start_run(run_name=run_name, nested=True):

            mlflow.tensorflow.autolog()
            x_train, y_train, x_val, y_val, x_test, y_test = load_processed_data(config)
            if channel_number != 6:
                logger.info(f"Running attribute {channel_number}")
                x_train = x_train[:, :, :, channel_number : channel_number + 1]
                x_val = x_val[:, :, :, channel_number : channel_number + 1]
                x_test = x_test[:, :, :, channel_number : channel_number + 1]
            else:
                logger.info("Running complete model")

            img_size_target = x_train.shape[1]
            number_of_channels = x_train.shape[-1]
            initial_number_of_filters = config["training"]["number_of_filters"]
            kernel_size = (
                config["training"]["kernel_size"],
                config["training"]["kernel_size"],
            )
            dropout_rate = config["training"]["dropout_rate"]
            logger.info(f"Image size: {img_size_target}")
            logger.info(f"Number of channels: {number_of_channels}")
            logger.info(f"Initial number of filters: {initial_number_of_filters}")
            logger.info(f"Kernel size: {kernel_size}")
            logger.info(f"Dropout rate: {dropout_rate}")

            inputLayer = Input((img_size_target, img_size_target, number_of_channels))
            outputLayer = build_model(
                inputLayer, initial_number_of_filters, kernel_size, dropout_rate
            )

            model = Model(inputLayer, outputLayer)

            METRICS = [
                tf.keras.metrics.TruePositives(name="tp"),
                tf.keras.metrics.FalsePositives(name="fp"),
                tf.keras.metrics.TrueNegatives(name="tn"),
                tf.keras.metrics.FalseNegatives(name="fn"),
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ]

            model.compile(
                loss=config["training"]["loss"],
                optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                metrics=METRICS,
            )

            early_stop = tf.keras.callbacks.EarlyStopping(
                patience=config["training"]["early_stop_patience"],
                monitor=config["training"]["early_stop_monitor"],
                mode="max",
                restore_best_weights=True,
            )

            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                "currentModel.h5",
                monitor=config["training"]["early_stop_monitor"],
                verbose=0,
                save_best_only=True,
            )
            logger.info(
                f"Early stop patience: {config['training']['early_stop_patience']}"
            )
            logger.info(
                f"Early stop monitor: {config['training']['early_stop_monitor']}"
            )

            # Callbacks list for Keras
            callbacks_list = [early_stop, checkpoint]

            BATCH_SIZE = config["training"]["batch_size"]
            EPOCHS = config["training"]["epochs"]
            logger.info(f"Batch size: {BATCH_SIZE}")
            logger.info(f"Epochs: {EPOCHS}")
            logger.info(f"Training samples: {x_train.shape[0]}")
            logger.info(f"Validation samples: {x_val.shape[0]}")
            logger.info(f"Test samples: {x_test.shape[0]}")

            history = model.fit(
                x_train,
                y_train,
                epochs=EPOCHS,
                callbacks=callbacks_list,
                batch_size=BATCH_SIZE,
                validation_data=(x_val, y_val),
                verbose=1,
            )

            logger.info("Training complete")

            log_test_metrics_and_history(x_test, y_test, model, history)
            logger.info("Test metrics logged")

            mlflow.log_artifact(script_dir.joinpath("outputs/train.log"))
            mlflow.log_artifact(f"{root_path}/outputs/.hydra/config.yaml")

            # try:
            #     os.remove(script_dir.joinpath("outputs/train.log"))
            #     # recreate the log file
            #     with open(script_dir.joinpath("outputs/train.log"), "w") as f:
            #         f.write("")
            # except:
            #     logger.error("Could not delete log file")
            #     pass

    logger.info(f"Trying to remove outputs directory: {root_path}/outputs")
    try:
        shutil.rmtree(f"{root_path}/outputs")
    except:
        logger.error("Could not remove outputs folder")
        pass


def log_test_metrics_and_history(x_test, y_test, model, history):
    test_metrics = model.evaluate(x_test, y_test)

    metrics_names = [
        "test_loss",
        "test_tp",
        "test_fp",
        "test_tn",
        "test_fn",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_auc",
    ]
    f1_score = (
        2
        * (test_metrics[-2] * test_metrics[-1])
        / (test_metrics[-2] + test_metrics[-1])
    )
    test_metrics = {metric: test_metrics[i] for i, metric in enumerate(metrics_names)}
    test_metrics["test_f1_score"] = f1_score
    mlflow.log_metrics(test_metrics)

    data_path = f"{root_path}/processed_data/"

    dump(history.history, open(f"{data_path}training_history.pkl", "wb"))
    mlflow.log_artifact(f"{data_path}training_history.pkl")


def load_processed_data(config: DictConfig):

    global root_path
    root_path = hydra.utils.get_original_cwd()
    data_path = f"{root_path}/processed_data/"

    if config["augmentation"]["use"] is True:
        logger.info("Loading augmented data")
        x_train = np.load(data_path + "x_train_augmented.npy")
        y_train = np.load(data_path + "y_train_augmented.npy")
    else:
        logger.info("Loading non-augmented data")
        x_train = np.load(data_path + "x_train_sampled.npy")
        y_train = np.load(data_path + "y_train_sampled.npy")

    x_val = np.load(data_path + "x_val_sampled.npy")
    y_val = np.load(data_path + "y_val_sampled.npy")
    x_test = np.load(data_path + "x_test_sampled.npy")
    y_test = np.load(data_path + "y_test_sampled.npy")

    return x_train, y_train, x_val, y_val, x_test, y_test


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
            logging.FileHandler(filename="train.log", mode="w"),
        ],
    )
    logger = logging.getLogger(__name__)

    main()