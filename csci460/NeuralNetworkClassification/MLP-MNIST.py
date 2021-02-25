# This code provides a simple example of a traditional 2-layer MLP
# on the MNIST character recognition problem

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf  # Ignore the warning

from csci460.Utils import DataGenerators as dg
from csci460.Utils import PlottingAndDisplay as plotting

### 1) Get the MNIST Data

# Keras lets us download the MNIST data directly ... that makes
# life pretty easy.
(trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()

# We want the instance space normalized between 0 and 1
trainX, testX = trainX / 255.0, testX / 255.0

# These are 60K images, each 28x28 of people writing integer numbers
# between 0 and 9.  You can see what the first looks like
plotting.ShowMNISTCharacter(trainX[0])

# So there are 10 possible class values, each corresponding to a digit:
print(trainY[0])


### 2) Build the ANN model -- a two-layer, feed-forward MLP using sigoid activation

# Start the model
model = tf.keras.models.Sequential()

# Flatten the input instances from 28x28 images to 1 big 784 element vector
model.add( tf.keras.layers.Flatten(input_shape=(28,28)) )

# Make a middle layer of 200 nodes, each using sigmoid activation
model.add( tf.keras.layers.Dense(200, activation="sigmoid") )

# Make the output size 10.  We're going to use a "hot ones"
# output representation.  Meaning, we'll have an output
# for each class, and the largest value will be selected.
model.add( tf.keras.layers.Dense(10) )

# Let's see what the model looks like:
print(model.summary())


### 3) Train the Model

# Setup sparse categorical crossentropy based on the logits
# That is, compute loss by comparing the logits coming out of
# the output layer to the true True/False using an entropy calc.
lossFunction = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

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

# Setup a model for predicting the probability by making a *new* model
# with a threshold unit that uses softmax
probModel = tf.keras.Sequential( [model, tf.keras.layers.Softmax()] )

# Let's just see what it does on the first test example
plotting.ShowMNISTCharacter(testX[0])  # Visualize instance image
print(testY[0])               # True class
probModel(testX[:1])          # Predicted class

# Show the testing accuracy over the un-thresholded model:
model.evaluate(testX, testY)


### 5) Plot the Training Performance

# Get the loss over time from the training history
accOverTime = trainingHistory.history['accuracy']
plotting.PlotTrainingPerformance(accOverTime, "MLP-MNIST-TrainingPerf.pdf")
