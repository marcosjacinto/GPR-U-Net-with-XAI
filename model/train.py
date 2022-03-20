from pickle import dump

import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from build_model import build_model


def main():
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

    testMetrics = model.evaluate(x_test, y_test)

    metrics = [
        "loss",
        "tp",
        "fp",
        "tn",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "auc",
        "dice",
    ]
    f1Score = (
        2 * (testMetrics[-2] * testMetrics[-1]) / (testMetrics[-2] + testMetrics[-1])
    )

    modelName = f"model_{kernelSize[0]}x{kernelSize[0]}_{initialNumberOfFilters}filters_complete"

    outputPath = "model/models/"

    with open(outputPath + modelName + "_description.txt", "w") as f:
        f.write("Model description:\n")
        f.write("This dataset contains the following features inside the array:\n")
        f.write(
            "GPR Section, Similarity, Energy, Instantaneous Frequency, Instantaneous Phase, Hilbert/Similarity\n"
        )
        f.write(f"Number of examples in the train dataset {x_train.shape[0]}\n")
        f.write(f"Number of examples in the validation dataset {x_val.shape[0]}\n")
        f.write(f"Number of examples in the test dataset {x_test.shape[0]}\n")
        f.write(f"Test metrics:\n")
        f.write(
            f"{metrics[0]} : {testMetrics[0]}; {metrics[1]} : {testMetrics[1]}; {metrics[2]} : {testMetrics[2]}\n"
        )
        f.write(
            f"{metrics[3]} : {testMetrics[3]}; {metrics[4]} : {testMetrics[4]}; {metrics[5]} : {testMetrics[5]}\n"
        )
        f.write(
            f"{metrics[6]} : {testMetrics[6]}; {metrics[7]} : {testMetrics[7]}; {metrics[8]} : {testMetrics[8]}\n"
        )
        f.write(f"f1 score : {f1Score}")

    model.save(outputPath + modelName + ".h5")
    dump(history.history, open(f"{outputPath + modelName}_history.pkl", "wb"))


def load_processed_data():

    data_path = "data/processed/"

    x_train = np.load(data_path + "x_train.npy")
    y_train = np.load(data_path + "y_train.npy")
    x_val = np.load(data_path + "x_val.npy")
    y_val = np.load(data_path + "y_val.npy")
    x_test = np.load(data_path + "x_test.npy")
    y_test = np.load(data_path + "y_test.npy")

    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == "__main__":

    main()
