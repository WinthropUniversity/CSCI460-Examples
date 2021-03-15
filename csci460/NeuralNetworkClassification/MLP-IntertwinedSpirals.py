import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf  # Ignore the warning

from csci460.Utils import DataGenerators as dg
from csci460.Utils import PlottingAndDisplay as plotting


### 1) Get the data
# Intertwined spirals are pretty hard to learn because of
# MLP's inuctive bias.  Consider both of these datasets:
trainX, testX, trainY, testY = dg.GenerateSpiralData()
#trainX, testX, trainY, testY = dg.GenerateCircleData()

# Make sure the data is scaled (roughly) between -1 and 1
scaleFactor = np.max(np.abs(trainX))
trainX = trainX / scaleFactor
testX  = testX / scaleFactor

# Let's setup the Y labels to be compatible with a "hot-ones" representation
trainY = tf.keras.utils.to_categorical(trainY)
testY  = tf.keras.utils.to_categorical(testY)


### 2) Build the ANN model -- a 6-layer, feed-forward MLP using
###    mostly rect. lin activation, though tanh for one layer
###    Try commenting some of these out ...

# Start the model
model = tf.keras.models.Sequential()

# Inputs are a 2D vector
model.add( tf.keras.layers.Input(shape=(2,)) )

# Make 1st middle layer of 200 nodes, each using rect. linear activation
model.add( tf.keras.layers.Dense(200, activation="relu") )

# Make 2nd middle layer of 200 nodes, each using rect. linear activation
model.add( tf.keras.layers.Dense(200, activation="tanh") )

# Make 3rd middle layer of 200 nodes, each using rect. linear activation
model.add( tf.keras.layers.Dense(200, activation="tanh") )

# Make 4th middle layer of 500 nodes, each using tanh activation
model.add( tf.keras.layers.Dense(500, activation="tanh") )

# Make the output size 2, for hot-ones, and use "softmax" activation
model.add( tf.keras.layers.Dense(2, activation="softmax") )

# Let's see what the model looks like:
print(model.summary())


### 3) Train the Model

# Setup binary crossentropy based on the logits.
# That is, compute loss by comparing the logits coming out of
# the output layer to the true True/False using an entropy calc.
lossFunction = tf.keras.losses.BinaryCrossentropy()

# Set the optimizer as a stochastic gradient descent method with
# a larning rate of 0.01
opt = tf.keras.optimizers.SGD(lr=0.01)

# Use an accuracy that compares the 2-node output to a binary class
acc = tf.keras.metrics.BinaryAccuracy()

# TF makes us "compile" the model, which gets it ready for training
#    * Use stochastic gradient descent as optimizer when training
#    * Use mean squared error as the loss function when training
model.compile( optimizer=opt, loss=lossFunction, metrics=[acc])

# Now perform the induction (i.e., learning, training, etc.)
# Remember the history of the training performance
trainingHistory = model.fit(trainX, trainY, epochs=3000)


### 4) Evaluate the Model on the Test Data

# Show the testing accuracy over the un-thresholded model:
print()
print("Test Set Eval:")
model.evaluate(testX, testY)


### 5) Show the Concept Map
plotting.Plot2DConceptMap( (np.max(trainX), np.min(trainX)),\
                           100,\
                           model,\
                           "../../data/svm-conceptmap.png")
