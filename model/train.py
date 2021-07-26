from pickle import dump

import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout

def build_model(inputLayer, numberOfFilters, kernelSize):
    # sampleSize -> sampleSize/2
    conv1 = Conv2D(numberOfFilters * 1, kernelSize, activation="relu", padding="same")(inputLayer)
    conv1 = Conv2D(numberOfFilters * 1, kernelSize, activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    # sampleSize/2 -> sampleSize/4
    conv2 = Conv2D(numberOfFilters * 2, kernelSize, activation="relu", padding="same")(pool1)
    conv2 = Conv2D(numberOfFilters * 2, kernelSize, activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    # sampleSize/4 -> sampleSize/8
    conv3 = Conv2D(numberOfFilters * 4, kernelSize, activation="relu", padding="same")(pool2)
    conv3 = Conv2D(numberOfFilters * 4, kernelSize, activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    # sampleSize/8 -> sampleSize/16
    conv4 = Conv2D(numberOfFilters * 8, kernelSize, activation="relu", padding="same")(pool3)
    conv4 = Conv2D(numberOfFilters * 8, kernelSize, activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    convm = Conv2D(numberOfFilters * 16, kernelSize, activation="relu", padding="same")(pool4)
    convm = Conv2D(numberOfFilters * 16, kernelSize, activation="relu", padding="same")(convm)

    # sampleSize/16 -> sampleSize/8
    deconv4 = Conv2DTranspose(numberOfFilters * 8, kernelSize, strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(numberOfFilters * 8, kernelSize, activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(numberOfFilters * 8, kernelSize, activation="relu", padding="same")(uconv4)

    # sampleSize/8 -> sampleSize/4
    deconv3 = Conv2DTranspose(numberOfFilters * 4, kernelSize, strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(numberOfFilters * 4, kernelSize, activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(numberOfFilters * 4, kernelSize, activation="relu", padding="same")(uconv3)

    # sampleSize/4 -> sampleSize/2
    deconv2 = Conv2DTranspose(numberOfFilters * 2, kernelSize, strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(numberOfFilters * 2, kernelSize, activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(numberOfFilters * 2, kernelSize, activation="relu", padding="same")(uconv2)

    # sampleSize/2 -> sampleSize
    deconv1 = Conv2DTranspose(numberOfFilters * 1, kernelSize, strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(numberOfFilters * 1, kernelSize, activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(numberOfFilters * 1, kernelSize, activation="relu", padding="same")(uconv1)

    #uconv1 = Dropout(0.5)(uconv1)
    outputLayer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    
    return outputLayer

data_path = 'data/processed/'

x_train = np.load(data_path + 'x_train.npy')
y_train = np.load(data_path + 'y_train.npy')
x_val = np.load(data_path + 'x_val.npy')
y_val = np.load(data_path + 'y_val.npy')
x_test = np.load(data_path + 'x_test.npy')
y_test = np.load(data_path + 'y_test.npy')

imgSizeTarget = x_train.shape[1]
numberOfChannels = x_train.shape[-1]
initialNumberOfFilters = 8
kernelSize = (5, 5)

inputLayer = Input((imgSizeTarget, imgSizeTarget, numberOfChannels))
outputLayer = build_model(inputLayer, initialNumberOfFilters, kernelSize)

model = Model(inputLayer, outputLayer)

METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]

model.compile(loss = "binary_crossentropy",
              optimizer= tf.keras.optimizers.Adam(lr=1e-3),
              metrics = METRICS)

early_stop = tf.keras.callbacks.EarlyStopping(patience = 30,
                                           monitor = 'val_accuracy',
                                           mode = 'max',
                                           restore_best_weights = False)

checkpoint = tf.keras.callbacks.ModelCheckpoint('currentModel.h5',
                                                monitor='val_accuracy',
                                                verbose=0, save_best_only=True,
                                                )

# Callbacks list for Keras
callbacks_list = [early_stop, checkpoint]

BATCH_SIZE = 256
EPOCHS = 2000

history = model.fit(
  x_train, y_train,
  epochs = EPOCHS,
  callbacks = callbacks_list,
  batch_size = BATCH_SIZE,
  validation_data = (x_val, y_val),
  verbose = 1
  )

testMetrics = model.evaluate(x_test, y_test)

metrics = ['loss', 'tp', 'fp', 'tn', 'fn', 'accuracy', 'precision', 'recall', 'auc', 'dice']
f1Score = 2 * (testMetrics[-2] * testMetrics[-1])/(testMetrics[-2] + testMetrics[-1] )

modelName = f'model_{kernelSize[0]}x{kernelSize[0]}_{initialNumberOfFilters}filters_complete'

outputPath = 'model/models/'

with open(outputPath + modelName + '_description.txt', 'w') as f:
  f.write('Model description:\n')
  f.write('This dataset contains the following features inside the array:\n')
  f.write('GPR Section, Similarity, Energy, Instantaneous Frequency, Instantaneous Phase, Hilbert/Similarity\n')
  f.write(f'Number of examples in the train dataset {x_train.shape[0]}\n')
  f.write(f'Number of examples in the validation dataset {x_val.shape[0]}\n')
  f.write(f'Number of examples in the test dataset {x_test.shape[0]}\n')
  f.write(f'Test metrics:\n')
  f.write(f'{metrics[0]} : {testMetrics[0]}; {metrics[1]} : {testMetrics[1]}; {metrics[2]} : {testMetrics[2]}\n')
  f.write(f'{metrics[3]} : {testMetrics[3]}; {metrics[4]} : {testMetrics[4]}; {metrics[5]} : {testMetrics[5]}\n')
  f.write(f'{metrics[6]} : {testMetrics[6]}; {metrics[7]} : {testMetrics[7]}; {metrics[8]} : {testMetrics[8]}\n')
  f.write(f'f1 score : {f1Score}')

model.save(outputPath + modelName +'.h5')
dump(history.history, open(f"{outputPath + modelName}_history.pkl", 'wb'))