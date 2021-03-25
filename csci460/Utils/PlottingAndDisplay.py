import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Use this to see what a character looks like.
def ShowMNISTCharacter(image):
  """
  Display the character in the image in ASCII form
  """
  nRows, nCols = np.shape(image)
  #
  for idx in range(nRows):
    for jdx in range(nCols):
      if image[idx,jdx] > 0.66:
        print('#', end='')  # Dark pixel
      elif image[idx,jdx] > 0.33:
        print('x', end='')  # Lighter pixel
      else:
        print('.', end='')  # No ink at all
    print()


def PlotTrainingPerformance(trainingPerformance, pdfFilename,\
                            xlab="epoch", ylab="Training Accuracy"):
  """
  Produce a PDF plot of the learning performance over time.  This function
  needs a vector of performances and a filename to save the PDF.  You can also
  relabel X and Y axes.
  """
  plt.plot(trainingPerformance)
  plt.xlabel(xlab)
  plt.ylabel(ylab)
  plt.savefig(pdfFilename)


def Plot2DConceptMap(dataRange, mapResolution, predictorFunction,\
                     pdfFilename="conceptmap.pdf", data=None):
  """
  This function produces a concept map for a potential 2D space.
  The map will query a grid of mapResolution by mapResolution,
  coloring all points in the class as black and leaving the rest
  white.
  """
  image = np.reshape(np.zeros(mapResolution**2), (mapResolution, mapResolution) )
  #xticks = np.linspace(min(df['X1']), max(df['X1']), mapResolution)
  #yticks = np.linspace(min(df['X2']), max(df['X2']), mapResolution)
  xticks = np.linspace(min(dataRange), max(dataRange), mapResolution)
  yticks = np.linspace(min(dataRange), max(dataRange), mapResolution)

  for x1dx in range(mapResolution):
    for x2dx in range(mapResolution):
      x1 = xticks[x1dx]
      x2 = yticks[x2dx]
      y = np.round( predictorFunction( np.array([[x1,x2]]) ) )

      # If there's just one output, then anything bigger than
      # one is in the class
      yClass = (np.max(y) > 0.5)

      # But maybe we're using softmax, in which case, let's
      # check the second dim to see if it's prob>0.5
      try:
          yClass = (y[0][1] > 0.5)
      except:
          pass

      # I added zero to turn bool into int
      image[x2dx,x1dx] = (0+yClass)

  plt.pcolor(xticks, yticks, image, cmap='gray')

  if (not data == None):
      trainX, trainY = data
      colors = {0:'lightblue', 1:'darkred'}
      df = pd.DataFrame({"X1":trainX[:,0],\
                         "X2":trainX[:,1],\
                         "Class":trainY})
      plt.scatter(df['X1'], df['X2'], color=df['Class'].map(colors))

  plt.savefig(pdfFilename)

  print()
  print("Go look at ", pdfFilename)


def SaveImage(image, filename, colormap=None):
    """
    Save the image into the specified filename.
    """
    plt.axis("off")

    if (colormap == None):
      plt.imshow(image)
    else:
      plt.imshow(image, cmap=colormap)

    plt.savefig(filename)


def ScreePlot(pca, pdfFilename="../../data/scree-plot.png"):
  """
  Given the PCA tool object that has been fit to data, plot the normalized
  absolute value of the Eigenvalues (explained variance) in order from 1st
  principle component to last.  This provides a sense of how many dimensions
  are needed to explain the majority of the variance in the data.
  """
  PC_values = np.arange(pca.n_components_) + 1
  plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2)
  plt.title('Scree Plot')
  plt.xlabel('Principal Component')
  plt.ylabel('Proportion of Variance Explained')
  plt.savefig(pdfFilename)


def FirstTwoPrinCompScatter(pca, df, pdfFilename="../../data/prcomp-plot.png"):
  """
  Given the PCA tool object that has been fit to data, and the data,
  transform the original data by each of the first two principle components,
  then scatterplot those data.
  """
  # Get the $m-by-d$ data matrix from the Pandas data frame
  X = df.to_numpy()

  # Create a d-by-2 matrix of the loading vectors for the first two princ. comp.
  PC = pca.components_[0:2].T

  print("shape(X): ", np.shape(X))
  print("shape(PC): ", np.shape(PC))
  # Multiply the m-by-d data matrix by the d-by-2 PC matrix.
  # The result is a *new* data matrix where each of the m points
  # now exist in a 2-dim vector space in which much of the variance is explained
  newX = np.matmul(X,PC)

  plt.scatter(newX[:,0], newX[:,1])
  plt.title('Scatter Plot of Data Transformed by First two PCs')
  plt.xlabel('PC 1, Trans')
  plt.ylabel('PC 2, Trans')
  plt.savefig(pdfFilename)
