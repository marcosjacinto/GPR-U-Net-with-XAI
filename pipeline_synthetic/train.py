import logging
import typing as t
from pathlib import Path
from pickle import dump

import mlflow
import numpy as np
import tensorflow as tf
from gpr_unet.model import build_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Input


def main():

    with mlflow.start_run(nested=True):

        mlflow.tensorflow.autolog()

        x_train, y_train, x_val, y_val, x_test, y_test = load_processed_data()

        imgSizeTarget = x_train.shape[1]
        numberOfChannels = x_train.shape[-1]
        initialNumberOfFilters = 8
        kernelSize = (5, 5)

        inputLayer = Input((imgSizeTarget, imgSizeTarget, numberOfChannels))
        outputLayer = build_model(inputLayer, initialNumberOfFilters, kernelSize)

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
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(lr=1e-3),
            metrics=METRICS,
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            patience=30, monitor="val_accuracy", mode="max", restore_best_weights=False
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            "currentModel.h5",
            monitor="val_accuracy",
            verbose=0,
            save_best_only=True,
        )

        # Callbacks list for Keras
        callbacks_list = [early_stop, checkpoint]

        BATCH_SIZE = 256
        EPOCHS = 2000

        history = model.fit(
            x_train,
            y_train,
            epochs=EPOCHS,
            callbacks=callbacks_list,
            batch_size=BATCH_SIZE,
            validation_data=(x_val, y_val),
            verbose=1,
        )

        log_test_metrics_and_history(x_test, y_test, model, history)

        mlflow.log_artifact(script_dir.joinpath("train.log"))


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

    dump(history.history, open(f"processed_data/training_history.pkl", "wb"))
    mlflow.log_artifact(f"processed_data/training_history.pkl")


def load_processed_data():

    data_path = "processed_data/"

    x_train = np.load(data_path + "x_train_augmented.npy")
    y_train = np.load(data_path + "y_train_augmented.npy")
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
        filename=script_dir.joinpath("train.log"),
        filemode="w",
    )
    logger = logging.getLogger(__name__)

    main()
