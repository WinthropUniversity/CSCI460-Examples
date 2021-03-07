# This code provides a simple example of an LSTM recurrent neural network
# to classify a time series of airline passangers.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf  # Ignore the warning
import sklearn

import csci460.Utils.PlottingAndDisplay as plotting

def CreateTimeWindows(ds, inputWindow=1, outputWindow=1):
    x, y = [], []
    for time in range(len(ds)-inputWindow-outputWindow):
        before = ds[time:(time+inputWindow),0]
        after  = ds[(time+inputWindow):(time+inputWindow+outputWindow),0]
        x.append(before)
        y.append(after)

    # Make these numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Make x into a tensor
    x = np.reshape(x, (x.shape[0], 1, x.shape[1]) )

    return x, y


## 1) Prepare the Data

# Use pandas to read the CSV data file in, but then strip just the values out
# of it because tensorflow doesn't read pandas dataframes directly.
dataframe = pd.read_csv("../../data/airline-passengers.csv", usecols=[1], engine="python")
dataset = dataframe.values.astype('float32')

# Scale values of the passenger counts between 0 and 1
minVal = np.min(dataset)
maxVal = np.max(dataset)
dataset = (dataset-minVal)/(maxVal - minVal)

# Break the timeseries into training (first 3/4s points) and testing (remainder)
trainingIdx = int(len(dataset) * 0.75)
trainAll = dataset[0:trainingIdx,:]
testAll = dataset[trainingIdx:len(dataset),:]

# Break up the time series into overlapping windows:
# Each example contains an input window number of observations
# And attempts to predict an output window number of observations
inputWindow = 10
outputWindow = 1
trainX, trainY = CreateTimeWindows(trainAll, inputWindow, outputWindow)
testX, testY = CreateTimeWindows(testAll, inputWindow, outputWindow)

print("Shape of TrainX=", tf.shape(trainX))
print("Shape of TrainY=", tf.shape(trainY))


## 2) Build the Model & Learn
model = tf.keras.models.Sequential()
model.add( tf.keras.layers.Input(shape=(1, inputWindow)) )
model.add( tf.keras.layers.LSTM(20) )

model.compile(loss="MSE", optimizer="adam", metrics="accuracy")
model.fit(trainX, trainY, epochs=2000, batch_size=5, verbose=2)


## 3) Evaluate the Model

# Evaluate on the test set
print()
print("Test Set Evaluation:")
testPredictions = model.evaluate(testX, testY)

# Make predictions over the test set, then put them back in the proper scale
testPredictions = model.predict(testX)
scaledPredictions = testPredictions * (maxVal - minVal) + minVal
scaledTestY = testY * (maxVal-minVal) + minVal

print()
print("Prediction vs. Actual, Test Set")
#print(np.shape(testPredictions), np.shape(testY))
n = np.shape(testPredictions)[0]
for idx in range(n):
    print(np.round(scaledPredictions[idx,0],1),\
          ", ", scaledTestY[idx,0])
