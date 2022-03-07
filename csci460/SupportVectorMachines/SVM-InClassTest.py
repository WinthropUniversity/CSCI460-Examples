import numpy as np
import sklearn.svm as svm
import sklearn.model_selection as sklearnmodels
import pandas as pd

### 1) Get the data
ds = pd.read_csv("../../data/processed.cleveland.data", header=None)
ds = ds[ ds[11] != '?' ]  # Ignore rows with '?' in col 11
ds = ds[ ds[12] != '?' ]  # Ignore rows with '?' in col 12

# Strip out the class from the instance, convert to numbers
Xpts = ds[range(13)].to_numpy(dtype='float32')
Ypts = ds[13].to_numpy(dtype='float32')

# Normalize each column of X into the range [0,1]
Xpts = Xpts / Xpts.max(axis=0)

# Convert Class to binary class:
Ypts = (Ypts > 0)+ 0

print("All X instances:")
print(Xpts)
print()

print("All Y class values:")
print(Ypts)
print()

# Divide up into training and testing
trainX, testX, trainY, testY = sklearnmodels.train_test_split(Xpts, Ypts, test_size=0.33)


### 2) Build and learn the model:
rbf_model = svm.SVC(kernel='poly', degree=100) #C=1, gamma=1)
rbf_fit   = rbf_model.fit(trainX, trainY)


### 3) Evaluate the model
numCorrect =  np.sum((0+ (rbf_fit.predict(trainX) > 0.5) ) == trainY)
numInstances = np.shape(trainY)[0]
print("Training Set Accuracy: ", float(numCorrect)/float(numInstances))

numCorrect =  np.sum((0+ (rbf_fit.predict(testX) > 0.5) ) == testY)
numInstances = np.shape(testY)[0]
print("Test Set Accuracy: ", float(numCorrect)/float(numInstances))

print()
print("Number of Support Vectors in Each class: ", rbf_fit.n_support_)
print("Learned coefficients: ", rbf_fit.dual_coef_)
print()
