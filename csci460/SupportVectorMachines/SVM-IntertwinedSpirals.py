import numpy as np

import sklearn.svm as svm

import csci460.Utils.DataGenerators as dg
import csci460.Utils.PlottingAndDisplay as plotting


### 1) Get the data
# Intertwined spirals are easier for RBF-based SVMs than MLPs
trainX, testX, trainY, testY = dg.GenerateSpiralData()

# Make sure the data is scaled (roughly) between -1 and 1
scaleFactor = np.max(np.abs(trainX))
trainX = trainX / scaleFactor
testX  = testX / scaleFactor


### 2) Build and learn the model:
rbf_model = svm.SVR(kernel='rbf', C=50, gamma=25)
rbf_fit   = rbf_model.fit(trainX, trainY)

### 3) Evaluate the model
numCorrect =  np.sum((0+ (rbf_fit.predict(testX) > 0.5) ) == testY)
numInstances = np.shape(testY)[0]
print("Test Set Accuracy: ", float(numCorrect)/float(numInstances))
print("Number of Support Vectors in Each class: ", rbf_fit.n_support_)
#print("Learned coefficients: ", rbf_fit.dual_coef_)
print()


### 4) Show the Concept Map
plotting.Plot2DConceptMap( (np.max(trainX), np.min(trainX)),\
                           100,\
                           rbf_fit.predict,\
                           "../../data/svm-conceptmap.png")
