import numpy as np

import sklearn.svm as svm

import csci460.Utils.DataGenerators as dg
import csci460.Utils.PlottingAndDisplay as plotting


### 1) Get the data
# Intertwined spirals are easier for RBF-based SVMs than MLPs
trainX, testX, trainY, testY = dg.GenerateLinearlySeparableData()

### 2) Build and learn the model:
model = svm.SVC(kernel='linear', C=1000,)
fit   = model.fit(trainX, trainY)

### 3) Evaluate the model
numCorrect =  np.sum((0+ (fit.predict(testX) > 0.5) ) == testY)
numInstances = np.shape(testY)[0]
print("Test Set Accuracy: ", float(numCorrect)/float(numInstances))
print("Number of Support Vectors in Each class: ", fit.n_support_)
#print("Learned coefficients: ", rbf_fit.dual_coef_)
print()


### 4) Show the Concept Map
plotting.Plot2DConceptMap( (np.max(trainX), np.min(trainX)),\
                           100,\
                           fit.predict,\
                           "../../data/svm-conceptmap.png",
                           data=(trainX, trainY))
