# This example uses TF to learn a simple single layer perceptron
# to implement logical one-bit adder.

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf  # Ignore the warning

from csci460.Utils import DataGenerators as dg
from csci460.Utils import PlottingAndDisplay as plotting


### 1)  Prepare the Data

trainX, trainY, testX, testY = dg.GenerateAdder()


### 2)  Build the Model

# Start with an ANN that is just a sequence of layers
model = tf.keras.models.Sequential()

# The first layer takes 2 inputs
model.add( tf.keras.layers.Input(shape=(3,)) )

# Hidden layer has 3 nodes
model.add( tf.keras.layers.Dense(6, activation="relu") )

# Output layer has 1 node
model.add( tf.keras.layers.Dense(2, activation="relu") )

# Let's see the summary
print(model.summary())


### 3) Train the Model

# TF makes us "compile" the model, which gets it ready for training
#    * Use stochastic gradient descent as optimizer when training
#    * Use mean squared error as the loss function when training
model.compile( optimizer="AdaMax", loss="MSE", metrics=["accuracy"] )

# Now perform the induction (i.e., learning, training, etc.)
# Remember the history of the training performance
trainingHistory = model.fit(trainX, trainY, epochs=10000)


### 4) Evaluate the Model on a Test Data
print()
print("Training Classification:")
print(np.round(model(trainX), 3))
print()
print("Training Truth:")
print(trainY)
print()

model.evaluate(testX, testY)


### 5) Plot the Training Performance

# Get the loss over time from the training history
lossOverTime = trainingHistory.history['loss']
plotting.PlotTrainingPerformance(lossOverTime,\
                                 "../../data/Adder-MNIST-TrainingPerf.pdf",\
                                 ylab="MSE")
