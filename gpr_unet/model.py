from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    concatenate,
    Dropout,
)


def build_model(input_layer, number_of_filters, kernel_size):
    # sampleSize -> sampleSize/2
    conv1 = Conv2D(
        number_of_filters * 1, kernel_size, activation="relu", padding="same"
    )(input_layer)
    conv1 = Conv2D(
        number_of_filters * 1, kernel_size, activation="relu", padding="same"
    )(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    # sampleSize/2 -> sampleSize/4
    conv2 = Conv2D(
        number_of_filters * 2, kernel_size, activation="relu", padding="same"
    )(pool1)
    conv2 = Conv2D(
        number_of_filters * 2, kernel_size, activation="relu", padding="same"
    )(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    # sampleSize/4 -> sampleSize/8
    conv3 = Conv2D(
        number_of_filters * 4, kernel_size, activation="relu", padding="same"
    )(pool2)
    conv3 = Conv2D(
        number_of_filters * 4, kernel_size, activation="relu", padding="same"
    )(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    # sampleSize/8 -> sampleSize/16
    conv4 = Conv2D(
        number_of_filters * 8, kernel_size, activation="relu", padding="same"
    )(pool3)
    conv4 = Conv2D(
        number_of_filters * 8, kernel_size, activation="relu", padding="same"
    )(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    convm = Conv2D(
        number_of_filters * 16, kernel_size, activation="relu", padding="same"
    )(pool4)
    convm = Conv2D(
        number_of_filters * 16, kernel_size, activation="relu", padding="same"
    )(convm)

    # sampleSize/16 -> sampleSize/8
    deconv4 = Conv2DTranspose(
        number_of_filters * 8, kernel_size, strides=(2, 2), padding="same"
    )(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(
        number_of_filters * 8, kernel_size, activation="relu", padding="same"
    )(uconv4)
    uconv4 = Conv2D(
        number_of_filters * 8, kernel_size, activation="relu", padding="same"
    )(uconv4)

    # sampleSize/8 -> sampleSize/4
    deconv3 = Conv2DTranspose(
        number_of_filters * 4, kernel_size, strides=(2, 2), padding="same"
    )(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(
        number_of_filters * 4, kernel_size, activation="relu", padding="same"
    )(uconv3)
    uconv3 = Conv2D(
        number_of_filters * 4, kernel_size, activation="relu", padding="same"
    )(uconv3)

    # sampleSize/4 -> sampleSize/2
    deconv2 = Conv2DTranspose(
        number_of_filters * 2, kernel_size, strides=(2, 2), padding="same"
    )(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(
        number_of_filters * 2, kernel_size, activation="relu", padding="same"
    )(uconv2)
    uconv2 = Conv2D(
        number_of_filters * 2, kernel_size, activation="relu", padding="same"
    )(uconv2)

    # sampleSize/2 -> sampleSize
    deconv1 = Conv2DTranspose(
        number_of_filters * 1, kernel_size, strides=(2, 2), padding="same"
    )(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(
        number_of_filters * 1, kernel_size, activation="relu", padding="same"
    )(uconv1)
    uconv1 = Conv2D(
        number_of_filters * 1, kernel_size, activation="relu", padding="same"
    )(uconv1)

    # uconv1 = Dropout(0.5)(uconv1)
    outputLayer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

    return outputLayer
