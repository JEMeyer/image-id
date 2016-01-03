import os
from PIL import Image
import numpy as np
import glob


def input_data(path):
  newPath = os.path.join(path,'*.jpg')
  filePaths = glob.glob(newPath)
  numImgs = len(filePaths)

  labels = np.empty(shape = (numImgs,4))
  features = np.empty(shape = (numImgs,16384))

  for i,imagePath in enumerate(filePaths):
    if 'perfecto' in imagePath:
      labelVec = [1,0,0,0]
    elif 'parejo' in imagePath:
      labelVec = [0,1,0,0]

    # if we load one image at a time, this is much faster than load_images()
    img = np.asarray(Image.open(imagePath)).ravel()

    features[i,:] = img
    labels[i,:] = labelVec
  return features,labels
