# This code provides a simple example of a convolutional deep network
# on the MNIST character recognition problem

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf  # Ignore the warning

import csci460.Utils.DataGenerators as dg
import csci460.Utils.PlottingAndDisplay as plotting


### 1) Get the MNIST Data

# Keras lets us download the MNIST data directly ... that makes
# life pretty easy.
(trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()

trainX= trainX.astype('float32')
testX = testX.astype('float32')

# We want the instance space normalized between 0 and 1
trainX = trainX / 255.0
testX  = testX / 255.0

# These are 60K images, each 28x28 of people writing integer numbers
# between 0 and 9.  But we need them to be setup as a tensor that is
# m by 28 by 28 by 1 (meaning there is 1 channel of data [grayscale]).
trainX = trainX.reshape( (-1, 28, 28, 1))
testX  = testX.reshape( (-1, 28, 28, 1))

# Let's setup the Y labels to be compatible with a "hot-ones" representation
trainY = tf.keras.utils.to_categorical(trainY)
testY  = tf.keras.utils.to_categorical(testY)



### 2) Build the ANN model -- a four-layer CNN

# Start the model
model = tf.keras.models.Sequential()

# Add a convolutional layer that takes in tensors 28x28x1, and reads
# it using 70 filters of size 3x3.  Make the activation function
# a rectified linear unit.
model.add( tf.keras.layers.Conv2D(70,\
                                  (3, 3),\
                                  activation="relu",\
                                  input_shape=(28, 28, 1)) )

# Add a layer that finds the maximum value in each 2x2 filter and reduces
# the image using this pooling method to a smaller image.
model.add( tf.keras.layers.MaxPooling2D((2, 2)) )

# Maybe we've got the feature selection we need, so flatten the image
# to a 1D vector.  After this, it's just straight-forward MLP.
model.add( tf.keras.layers.Flatten() )

# Make a middle layer of 70 nodes, each using rectified linear activation
model.add( tf.keras.layers.Dense(70, activation="relu") )

# Make the output size 10. Let's softmax the activation so
# we get probabilities in the end.
model.add( tf.keras.layers.Dense(10, activation="softmax") )

# Let's see what the model looks like:
print(model.summary())


### 3) Train the Model

# Setup categorical crossentropy.  I can do this rather than the
# sparse from-logits because I've reshaped the Y vectors to be hot-ones.
lossFunction = tf.keras.losses.CategoricalCrossentropy()

# Set the optimizer as a stochastic gradient descent method with
# a larning rate of 0.01
opt = tf.keras.optimizers.SGD(lr=0.01)

# Set the model for training
#  * Use stochastic gradient descent for optimization
#  * Compute loss using sparse categorical cross entropy
#  * Report the performance during learning using accuracy
model.compile( optimizer=opt, loss=lossFunction, metrics=['accuracy'])

# Perform the induction
trainingHistory = model.fit(trainX, trainY, epochs=10)


### 4) Evaluate the Model on the Test Data

# Show the testing accuracy over the un-thresholded model:
model.evaluate(testX, testY)


### 5) Plot the Training Performance

# Get the loss over time from the training history
accOverTime = trainingHistory.history['accuracy']
plotting.PlotTrainingPerformance(accOverTime, "Conv2D-MNIST-TrainingPerf.pdf")
