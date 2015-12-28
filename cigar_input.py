import tensorflow.python.platform
import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def read_cigar(path):
  '''
  imput:
    path: a path to a dir or image file
  
  returns:
    images: a numpy array of arrays of images
  '''
  # Create a queue that produces the filenames to read
  if os.path.isdir(path):
    filenames = [os.path.join(path,filename) for filename in os.listdir(path)]
  elif os.path.isfile(path):
    filenames= [path]
  filename_queue = tf.train.string_input_producer(filenames)

  # our tensorflow ops read images, getting filenames from the filename_queue
  reader = tf.WholeFileReader()
  key, value = reader.read(filename_queue)
  img = tf.image.decode_jpeg(value)
  init_op = tf.initialize_all_variables()

  # initialize ops and run the session
  with tf.Session() as sess:
    sess.run(init_op)
    # start populating the filename queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    images=[]
    for i in range(len(filenames)):
      image = img.eval()
      print('image.shape '+ str(image.shape))
      plot_image(image)
      images.append(image)
    coord.request_stop()
    coord.join(threads)

  images = np.asarray(images)
  print('total images read: ' + str(len(images)))
  return(images)

def plot_image(image):
  '''
  given an image as an array, plot it if it's RGB
  '''
  if image.shape[2] == 3:
    fig = plt.figure()
    plt.imshow(image)
    plt.show()
  elif image.shape[2] == 1:
    print('GRAYSCALE IMAGE... dont know how to show')

if __name__ == '__main__':
  import sys
  path = sys.argv[1]

  imageData = read_cigar(path)
