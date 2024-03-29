# This code provides a simple example of a traditional 2-layer MLP
# on the MNIST character recognition problem

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf  # Ignore the warning

import csci460.Utils.DataGenerators as dg
import csci460.Utils.PlottingAndDisplay as plotting

def BuildEncoder(latentDim, imageWidth, imageHeight):
    inputLayer = tf.keras.layers.Input(shape=(imageWidth, imageHeight, 3))
    encLayer1 = tf.keras.layers.Conv2D(100, (3, 3),\
                                      #kernel_regularizer=tf.keras.regularizers.l1(0.01),\
                                      #activity_regularizer=tf.keras.regularizers.l2(0.01),\
                                      activation="relu")
    encLayer2 = tf.keras.layers.MaxPooling2D((2, 2))
    encLayer3 = tf.keras.layers.Conv2D(50, (3, 3), activation="relu")
    encLayer4 = tf.keras.layers.MaxPooling2D((4, 4))
    encFlatten = tf.keras.layers.Flatten()
    encDropout = tf.keras.layers.Dropout(0.05)
    #encLayer5 = tf.keras.layers.Dense(75,\
    #                                  kernel_regularizer=tf.keras.regularizers.l1(0.01),\
    #                                  activity_regularizer=tf.keras.regularizers.l2(0.01),\
    #                                  activation="relu")
    encLayer6 = tf.keras.layers.Dense(latentDim, activation="relu")
    latentFeatureLayer = tf.keras.layers.Dense(latentDim, activation="relu")

    # Build the actual model
    model = tf.keras.models.Sequential([inputLayer,\
                                        encLayer1,\
                                        encLayer2,\
                                        encLayer3,\
                                        encLayer4,\
                                        encFlatten,\
                                        encDropout,\
                                        #encLayer5,\
                                        encLayer6,\
                                        latentFeatureLayer])

    return model


def BuildDecoder(latentDim, imageWidth, imageHeight):
    colorChannels=3

    upScale1 = 2
    upScale2 = 1
    subWidth1 = int(imageWidth / (upScale1*upScale2))
    subHeight1 = int(imageHeight / (upScale1*upScale2))
    subWidth2 = int(imageWidth/upScale2)
    subHeight2 = int(imageHeight/upScale2)

    latentInputLayer = tf.keras.layers.Input(shape=(latentDim,))
    decLayer1 = tf.keras.layers.Dense(colorChannels*subWidth1*subHeight1,\
                                      #kernel_regularizer=tf.keras.regularizers.l1(0.01),\
                                      #activity_regularizer=tf.keras.regularizers.l2(0.01),\
                                      activation="relu")
    decDropout = tf.keras.layers.Dropout(0.05)
    decLayer2 = tf.keras.layers.Reshape( (subWidth1, subHeight1, colorChannels) )
    decLayer3 = tf.keras.layers.UpSampling3D( (upScale1, upScale1, 1) )
    outputLayer = tf.keras.layers.Conv2D(colorChannels, (4, 4), padding="same",\
    #                                  kernel_regularizer=tf.keras.regularizers.l1(0.01),\
    #                                  activity_regularizer=tf.keras.regularizers.l2(0.01),\
                                      activation="sigmoid")
    #decLayer5 = tf.keras.layers.UpSampling3D( (upScale2, upScale2, 1) )
    #outputLayer = tf.keras.layers.Conv2D(colorChannels, (3, 3), padding="same", activation="sigmoid")

    # Build the actual model
    model = tf.keras.models.Sequential([latentInputLayer,\
                                        decLayer1,\
                                        decDropout,\
                                        decLayer2,\
                                        decLayer3,\
                                        #decLayer4,\
                                        #decLayer5,\
                                        outputLayer])

    return model

latentDim = 200

### 1) Get the Data

# Read the first image and make a training dataset for it, call it "A"
rawImage = tf.keras.preprocessing.image.load_img("rpw-face.png", target_size=(100,100))
imageA = tf.keras.preprocessing.image.img_to_array(rawImage)
print(tf.shape(imageA))

width = 100
height = 100
imagesA = tf.reshape( [imageA], (1,width,height,3))
trainA = tf.cast(imagesA, 'float32') / 255.0

print("YAY!! One image loaded!!")

# Read the second image and make a training set for it, call it "B"
rawImage = tf.keras.preprocessing.image.load_img("knuth-face.png", target_size=(100,100))
imageB = tf.keras.preprocessing.image.img_to_array(rawImage)
imagesB = tf.reshape( [imageB], (1,width,height,3))
trainB = tf.cast(imagesB, 'float32') / 255.0
(widthB, heightB, _) = tf.shape(imageB)
print("Image B is: ", widthB, "by", heightB)

print("YAY!! Another image loaded!!")


# Assume A and B are the same dimesions for this silly, simple example

# We actually don't care at all about the labels (Y); we wont' use them


### 2a) Build Encoder (shared) representation
encoderModel = BuildEncoder(latentDim, width, height)
print()
print("ENCODER MODEL:")
print(encoderModel.summary())


### 2b) Build the two decoder representations
decoderModelA = BuildDecoder(latentDim, width, height)
decoderModelB = BuildDecoder(latentDim, width, height)

# Let's see what the A model looks like (B should be the same)
print()
print("DECODER A MODEL:")
print(decoderModelA.summary())


### 2b) Build the autoencoders for both:  Use the shared encoder in
###     both cases, but separate decoders
layersList = [ tf.keras.layers.Input(shape=(width, height, 3)) ]
layersList.extend( encoderModel.layers )   # Add all encoder layers
layersList.extend( decoderModelA.layers )  # Add all decoderA layers but the input
autoencoderModelA = tf.keras.models.Sequential(layersList)

layersList = [ tf.keras.layers.Input(shape=(width, height, 3)) ]
layersList.extend( encoderModel.layers )   # Add all encoder layers
layersList.extend( decoderModelB.layers )  # Add all decoderB layers but the input
autoencoderModelB = tf.keras.models.Sequential(layersList)

print()
print("FULL AUTOECODER A MODEL:")
print(autoencoderModelA.summary())
print()


### 3) Train the Models

# Set the model for training
#  * Use the ADAptive Momementum method for optimization
#  * Compute loss using mean absolute error
#  * Report the performance during learning using accuracy
autoencoderModelA.compile( optimizer="Adam", loss="MAE", metrics=['accuracy'])
autoencoderModelB.compile( optimizer="Adam", loss="MAE", metrics=['accuracy'])

# Perform the induction by alternative between the two networks.
# Each pass, the shared encoder is updated, as is the specific decoder
epochsPerIteration=4
accA = 0
accB = 0
lossA = -1
lossB = -1
for epochIdx in range(10000):
    #print("Overall Epoch:", epochIdx*epochsPerIteration)
    if (lossA < lossB):
      historyA = autoencoderModelA.fit(trainA, trainA, epochs=1, verbose=0)
      historyB = autoencoderModelB.fit(trainB, trainB, epochs=epochsPerIteration, verbose=0)
      accA = historyA.history['accuracy'][-1]
      lossA = historyA.history['loss'][-1]
      accB = historyB.history['accuracy'][-1]
      lossB = historyB.history['loss'][-1]
      print(epochIdx, "::  A", np.round(accA,4), np.round(lossA,4), "  :: *B", np.round(accB,4), np.round(lossB,4))
    else:
      historyA = autoencoderModelA.fit(trainA, trainA, epochs=epochsPerIteration, verbose=0)
      historyB = autoencoderModelB.fit(trainB, trainB, epochs=1, verbose=0)
      accA = historyA.history['accuracy'][-1]
      lossA = historyA.history['loss'][-1]
      accB = historyB.history['accuracy'][-1]
      lossB = historyB.history['loss'][-1]
      print(epochIdx, ":: *A", np.round(accA,4), np.round(lossA,4), "  ::  B", np.round(accB,4), np.round(lossB,4))

    if np.min([accA, accB]) > 0.98:
      break

#encoderModel.save("encoder.h5")
#decoderModelA.save("decoderA.h5")
#decoderModelB.save("decoderB.h5")

#encoderModel = tf.keras.models.load_model("encoder.h5")
#decoderModelA = tf.keras.models.load_model("decoderA.h5")
#decoderModelB = tf.keras.models.load_model("decoderB.h5")

### 6) The Ol'Switcheroo
print()
print("Fake!!")

# Use image A's newly learned decoder, but feed it the latent image for image B,
# then vice-versa
latentImageA = encoderModel(trainA)
latentImageB = encoderModel(trainB)

rebuiltImageMe = decoderModelA(latentImageA)
rebuiltImageMatt = decoderModelA(latentImageB)
rebuiltExampleAB = decoderModelA(latentImageB)
rebuiltExampleBA = decoderModelB(latentImageA)

# Save the Test set image 1, original
plotting.SaveImage(tf.saturate_cast(255.0*rebuiltImageMe[0], dtype = tf.uint8),\
                   "../../data/nofakeAA.png")
plotting.SaveImage(tf.saturate_cast(255.0*rebuiltImageMatt[0], dtype = tf.uint8),\
                   "../../data/nofakeBB.png")
plotting.SaveImage(tf.saturate_cast(255.0*rebuiltExampleAB[0], dtype = tf.uint8),\
                   "../../data/deepfakeAB.png")
plotting.SaveImage(tf.saturate_cast(255.0*rebuiltExampleBA[0], dtype = tf.uint8),\
                   "../../data/deepfakeBA.png")
#plotting.SaveImage(tf.saturate_cast(255.0*rebuiltExamples[1], dtype = tf.uint8),\
#                   "../../data/mattsmith-rebuilt.png")
