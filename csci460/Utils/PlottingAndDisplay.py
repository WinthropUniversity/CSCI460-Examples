import matplotlib.pyplot as plt
import numpy as np


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
                     pdfFilename="conceptmap.pdf"):
  """
  This function produces a concept map for a potential 2D space.
  The map will query a grid of mapResolution by mapResolution,
  coloring all points in the class as black and leaving the rest
  white.
  """
  maxRange = np.ceil(max(dataRange))
  minRange = np.floor(min(dataRange))

  image = np.reshape(np.zeros(mapResolution**2), (mapResolution, mapResolution) )

  blah = False
  for x1dx in range(mapResolution):
    for x2dx in range(mapResolution):
      x1 = (maxRange - minRange) * (x1dx/mapResolution) + minRange
      x2 = (maxRange - minRange) * (x2dx/mapResolution) + minRange
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
      image[x1dx,x2dx] = (0+yClass)

  plt.imshow(image, cmap='gray')
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
