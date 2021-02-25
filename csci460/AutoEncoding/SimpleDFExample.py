# This code provides a simple example of a traditional 2-layer MLP
# on the MNIST character recognition problem

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf  # Ignore the warning

import csci460.Utils.DataGenerators as dg
import csci460.Utils.PlottingAndDisplay as plotting


### 1) Get the Data

image = tf.image.decode_png(tf.io.read_file("../../data/rpw-pic.png"), channels=3)
images = tf.reshape(image, (1,630,630,3))
trainX = tf.cast(images, 'float32') / 255.0

# We actually don't care at all about the labels (Y); we wont' use them


## How many dimensions should the latent attribute vector be?
latentDim = 200


### 2a) Build Encoder (alone) representation

# Create the encoder layers
inputLayer = tf.keras.layers.Input(shape=(630, 630, 3))
encLayer1 = tf.keras.layers.Conv2D(100, (3, 3), activation="relu")
encLayer2 = tf.keras.layers.MaxPooling2D((5, 5))
encFlatten = tf.keras.layers.Flatten()
encLayer3 = tf.keras.layers.Dense(100, activation="relu")
latentFeatureLayer = tf.keras.layers.Dense(latentDim, activation="relu")

# Build the actual model
encoderModel = tf.keras.models.Sequential([inputLayer,\
                                           encLayer1,\
                                           encLayer2,\
                                           encFlatten,\
                                           #encLayer3,\
                                           latentFeatureLayer])
print()
print("ENCODER MODEL:")
print(encoderModel.summary())


### 2b) Build the decoder (alone) representation

# Create the decoder layers
latentInputLayer = tf.keras.layers.Input(shape=(latentDim,))
decLayer1 = tf.keras.layers.Dense(210*210*3, activation="relu")
decLayer2 = tf.keras.layers.Reshape( (210, 210, 3))
decLayer3 = tf.keras.layers.UpSampling3D( (3, 3, 1))
outputLayer = tf.keras.layers.Conv2D(3, (4, 4), padding="same", activation="sigmoid")


# Build the actual model
decoderModel = tf.keras.models.Sequential([latentInputLayer,\
                                           decLayer1,\
                                           decLayer2,\
                                           decLayer3,\
                                           outputLayer])

# Let's see what the model looks like:
print()
print("DECODER MODEL:")
print(decoderModel.summary())

### 2b) Build the whole autoencoder.  You must re-use the layers created above
autoencoderModel = tf.keras.models.Sequential([inputLayer,\
                                               encLayer1,\
                                               encLayer2,\
                                               encFlatten,\
                                               #encLayer3,\
                                               latentFeatureLayer,\
                                               decLayer1,\
                                               decLayer2,\
                                               decLayer3,\
                                               outputLayer])

print()
print("FULL AUTOECODER MODEL:")
print(autoencoderModel.summary())
print()


print("Shape of training set: ", tf.shape(trainX))
print("Shape of expected input: ", autoencoderModel.layers[0].input_shape)
print()

### 3) Train the Model

# Set the model for training
#  * Use AdaM for optimization
#  * Compute loss using binary cross entropy
#  * Report the performance during learning using accuracy
autoencoderModel.compile( optimizer="Adam", loss="MAE", metrics=['accuracy'])

# Perform the induction.  This looks a little weird ... so:
#  a) We are training the network to reproduce its own input.  Thus
#     each example is also its own truth
#  b) We're apply validation, but we may as use the test set because
#     the test set isn't needed for unspervised learning
trainingHistory = autoencoderModel.fit(trainX, trainX, epochs=200)


### 4) Let's Take a Look


# Let's just see what it does on the first test example
rebuiltExamples = autoencoderModel(trainX)
#originalImage = tf.reshape(testX[0], (28, 28))
#rebuiltImage  = tf.reshape(rebuiltExamples[0], (28, 28))

# Use Matplotlib to produce a PDF
import matplotlib.pyplot as plt

print("Pixel (315,315) in image 1:", rebuiltExamples[0][314][314])

# Save the Test set image 1, original
plotting.SaveImage(tf.saturate_cast(255.0*rebuiltExamples[0], dtype = tf.uint8),\
                   "../../data/rpw-rebuilt.png")
