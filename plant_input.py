import os
from PIL import Image
import numpy as np
import glob


def input_data(path):
    newPath = os.path.join(path,'*.jpg')
    filePaths = glob.glob(newPath)
    numImgs = len(filePaths)

    # get numPixels from one image, all images must be square and same dimension
    numPixels = np.asarray(Image.open(filePaths[0])).shape[0]

    labels = np.empty(shape = (numImgs,2))
    features = np.empty(shape = (numImgs,numPixels**2))

    for i,imagePath in enumerate(filePaths):
        if 'pinoak' in imagePath:
            labelVec = [1,0]
        elif 'sugarmaple' in imagePath:
            labelVec = [0,1]

        # if we load one image at a time, this is much faster than load_images()
        img = np.asarray(Image.open(imagePath)).ravel()

        features[i,:] = img
        labels[i,:] = labelVec
    return features,labels
