import tensorflow as tf
import cigar_input as ci
import numpy as np
import sys, random

def get_random_subset(batchSize, features, labels):
  '''
  Summary: Take a random subset of our training data for 1 epoch of training
  Input:
    batchSize: Integer. How many samples are requested
  Output: Pair of features and labels of size batchSize
  '''
  # Shuffle indexes
  indexes = [i for i in range(features.shape[0])]
  random.shuffle(indexes)
  # Get first batchSize number of random indexes
  tempIndexes = indexes[:batchSize]
  # Return matching pairs of features and labels based on random indexes
  return features[tempIndexes,:], labels[tempIndexes,:]
  

def get_test_train_sets(path):
  # Load in all labels and all features from images in folder
  allFeatures,allLabels = ci.main(path)

  # Get total number of images, as well as percentge of pictures for testing
  totalRecords = allLabels.shape[0]
  testPercentage = .10
  testRecordCount = int(totalRecords * testPercentage)

  # Split into test and training sets (labels and features)
  indexes = range(totalRecords)
  indexes = [i for i in indexes]
  random.shuffle(indexes)

  # Make testing set
  testIndexes = indexes[:testRecordCount]
  testFeatures, testLabels = allFeatures[testIndexes,:], allLabels[testIndexes,:]

  # Make training set
  trainFeatures = np.delete(allFeatures, testIndexes,axis=0)
  trainLabels = np.delete(allLabels, testIndexes,axis=0)

  return trainFeatures, trainLabels, testFeatures, testLabels



if __name__ == "__main__":
  # Path to folder containing all data  
  path = sys.argv[1]

  trainFeatures,trainLabels,testFeatures,testLabels = get_test_train_sets(path)

  # Define our tensor to hold images. 16384 = pixels, None indicates
  # we can hold any number of images
  x = tf.placeholder(tf.float32, [None, 16384])
  # Create the Variables for the weights and biases
  W = tf.Variable(tf.zeros([16384, 4]))
  b = tf.Variable(tf.zeros([4]))
  # This is our prediction output
  y = tf.nn.softmax(tf.matmul(x, W) + b)
  # This will be our correct answers/output
  y_ = tf.placeholder(tf.float32, [None, 4])
  # Calculate cross entropy by taking the negative sum of our correct values
  # multiplied by the log of our predictions
  cross_entropy = -tf.reduce_sum(y_*tf.log(y))
  # This is training the NN with backpropagation
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
  # Initializes all the variables we made above
  init = tf.initialize_all_variables()

  # Create and launch the session
  sess = tf.Session()
  sess.run(init)

  # Train. In this case, 1000 times
  for i in range(100):
    # Feeds in 1500 random images each time for training (stochastic training)
    # batch_xs are our images (pixels), batch_ys are our correct outputs
    batch_xs, batch_ys = get_random_subset(100, trainFeatures, trainLabels)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # tf.argmax gives the index of the highest entry in a tensor along some axis
  # argmax (y, 1) gives the label our model said was right, argmax(y_, 1) is the
  # correct label. tf.equal sees if these two are the same
  # This returns a list of booleans saying if it's true or false (predicted or
  # not)
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

  # False is 0 and True is 1, what was our average?
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

  # How well did we do?
  print(sess.run(accuracy, feed_dict={x: testFeatures, y_: testLabels}))

  # Close the session (self documenting code)
  sess.close()
