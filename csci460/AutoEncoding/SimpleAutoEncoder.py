# This code provides a simple example of a traditional 2-layer MLP
# on the MNIST character recognition problem

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf  # Ignore the warning

import csci460.Utils.DataGenerators as dg
import csci460.Utils.PlottingAndDisplay as plotting



### 1) Get the MNIST Data

# Keras lets us download the MNIST data directly ... that makes
# life pretty easy.
(trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()

# These are 60K images, each 28x28 of people writing integer numbers
# between 0 and 9.

# We want the instance space normalized between 0 and 1, reshape
# the 28x28 images to be 784x1 (flatten them)
trainX = trainX.astype('float32') / 255.0
plotting.ShowMNISTCharacter(trainX[0]) # See what it looks like before we flatten
trainX = trainX.reshape((len(trainX), np.prod(trainX.shape[1:])))

testX  = testX.astype('float32') / 255.0
testX  = testX.reshape((len(testX), np.prod(testX.shape[1:])))

# We actually don't care at all about the labels (Y); we wont' use them


## How many dimensions should the latent attribute vector be?
latentDim = 2


### 2a) Build Encoder (alone) representation
###       784x1 inputs  ->
###         dense layer with 128 outputs  ->
###         dense layer with latentDim ouputs (latent feature vector)

# Create the encoder layers
inputLayer = tf.keras.layers.Input(shape=(784,))
encHiddenLayer = tf.keras.layers.Dense(128, activation="relu")
latentFeatureLayer = tf.keras.layers.Dense(latentDim, activation="relu")

# Build the actual model
encoderModel = tf.keras.models.Sequential([inputLayer,\
                                           encHiddenLayer,\
                                           latentFeatureLayer])
print()
print("ENCODER MODEL:")
print(encoderModel.summary())


### 2b) Build the decoder (alone) representation
###       latentDimx1 inputs (latent feature vector)  ->
###         dense layer with 128 outputs  ->
###         dense layer with 784 ouputs (output image vector)

# Create the decoder layers
latentInputLayer = tf.keras.layers.Input(shape=(latentDim,))
decHiddenLayer = tf.keras.layers.Dense(128, activation="relu")
outputLayer = tf.keras.layers.Dense(784, activation="relu")

# Build the actual model
decoderModel = tf.keras.models.Sequential([latentInputLayer,\
                                           decHiddenLayer,\
                                           outputLayer])

# Let's see what the model looks like:
print()
print("DECODER MODEL:")
print(decoderModel.summary())


### 2b) Build the whole autoencoder.  You must re-use the layers created above
autoencoderModel = tf.keras.models.Sequential([inputLayer,\
                                               encHiddenLayer,\
                                               latentFeatureLayer,\
                                               decHiddenLayer,\
                                               outputLayer])

print()
print("FULL AUTOECODER MODEL:")
print(autoencoderModel.summary())
print()

### 3) Train the Model

# Set the model for training
#  * Use AdaM for optimization
#  * Compute loss using binary cross entropy
#  * Report the performance during learning using accuracy
autoencoderModel.compile( optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

# Perform the induction.  This looks a little weird ... so:
#  a) We are training the network to reproduce its own input.  Thus
#     each example is also its own truth
#  b) We're apply validation, but we may as use the test set because
#     the test set isn't needed for unspervised learning
trainingHistory = autoencoderModel.fit(trainX, trainX,\
                                       epochs=500, batch_size=256, shuffle=True,\
                                       validation_data=(testX,testX))


### 4) Let's Take a Look

# What does the latent feature vector look like for testX[0] ?
print(encoderModel([testX[0:]]))

# Let's just see what it does on the first test example
rebuiltExamples = autoencoderModel(testX)
originalImage = tf.reshape(testX[0], (28, 28))
rebuiltImage  = tf.reshape(rebuiltExamples[0], (28, 28))

print()
print("ORIGINAL IMAGE:")
plotting.ShowMNISTCharacter(originalImage)  # Visualize instance image

print()
print("REBUILT IMAGE:")
plotting.ShowMNISTCharacter(rebuiltImage)  # Visualize instance image

# Use Matplotlib to produce a PDF
import matplotlib.pyplot as plt

# Get all latent vectors from the test set... then scatter plot 2 dim of them
latentVector = encoderModel(testX)
x = latentVector[0:-1,0].numpy()
y = latentVector[0:-1,1].numpy()
plt.plot(x, y, 'o', color='black')
plt.xlabel("LA1")
plt.ylabel("LA2")
plt.savefig("LatentFeatureScatterPlot.pdf")

# Save the Test set image 1, original
plotting.SaveImage(originalImage,\
                   "../../data/MNIST-OriginalImage.png",\
                   "gray")

plotting.SaveImage(rebuiltImage,\
                   "../../data/MNIST-RebuiltImage.png",\
                   "gray")
