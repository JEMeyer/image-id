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
  sess = tf.InteractiveSession()
  # Path to folder containing all data  
  path = sys.argv[1]

  trainFeatures,trainLabels,testFeatures,testLabels = get_test_train_sets(path)

  # Define our tensor to hold images. 16384 = pixels, None indicates
  # we can hold any number of images
  x = tf.placeholder(tf.float32, [None, 16384])
  # Create the Variables for the weights and biases
  W = tf.Variable(tf.zeros([16384, 4]))
  b = tf.Variable(tf.zeros([4]))
  # Create and launch the session
  sess.run(tf.initialize_all_variables())
  # This is our prediction output
  y = tf.nn.softmax(tf.matmul(x, W) + b)
  # This will be our correct answers/output
  y_ = tf.placeholder(tf.float32, [None, 4])

  # Two functions to create weights and biases. Slightly positive bias due to
  # using ReLU neurons. This is all to avoid noise for symmetry breaking
  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  # Our convolutions uses a stride of one and are zero padded so that the output
  # is the same size as the input. Using 2x2 blocks for pooling
  def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

  # The first two dimensions are the patch size, the next is the number of input
  # channels, and the last is the number of output channels.
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])

  # Reshape image to 4d tensor. The second and third dimensions corresponding to
  # image width and height, and the final dimension corresponding to the number
  # of color channels
  # -1 in size calculates what the shape should be to have other values constant
  x_image = tf.reshape(x, [-1,124,124,1])

  # Convolute, then apply ReLU, then max_pool
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolution layer
  # This layer will have 65 features with a 5x5 patch
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])

  # Convolute, then apply ReLU, then max_pool
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  # Images are now 62x62. Final layer is fully connected
  W_fc1 = weight_variable([62 * 62 * 64, 16384])
  b_fc1 = bias_variable([16384])

  # We reshape the tensor from the pooling layer into a batch of vectors,
  # multiply by a weight matrix, add a bias, and apply a ReLU
  h_pool2_flat = tf.reshape(h_pool2, [-1, 62*62*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Apply dropout
  # Placeholder holds probability of a neuron getting kept
  keep_prob = tf.placeholder("float")
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Softmax layer (final layer)
  W_fc2 = weight_variable([16384, 4])
  b_fc2 = bias_variable([4])

  y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

  # We will replace (compared to mnistBeg.py) the steepest gradient descent
  # optimizer with the more sophisticated ADAM optimizer; we will include the
  # additional parameter keep_prob in feed_dict to control the dropout rate
  # Calculate cross entropy by taking the negative sum of our correct values
  # multiplied by the log of our predictions
  cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
  # This is training the NN with AdomOptimizer  
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  sess.run(tf.initialize_all_variables())
  for i in range(20000):
    batch_xs, batch_ys = get_random_subset(100, trainFeatures, trainLabels)
    if i%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x:batch_xs, y_: batch_ys, keep_prob: 1.0})
      print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

  testXs, testYs 
  print("test accuracy %g"%accuracy.eval(feed_dict={
    x: testFeatures, y_: testLabels, keep_prob: 1.0}))

# 
# 
#   # Train. In this case, 1000 times
#   for i in range(100):
#     # Feeds in 1500 random images each time for training (stochastic training)
#     # batch_xs are our images (pixels), batch_ys are our correct outputs
#     batch_xs, batch_ys = get_random_subset(100, trainFeatures, trainLabels)
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# 
#   # tf.argmax gives the index of the highest entry in a tensor along some axis
#   # argmax (y, 1) gives the label our model said was right, argmax(y_, 1) is the
#   # correct label. tf.equal sees if these two are the same
#   # This returns a list of booleans saying if it's true or false (predicted or
#   # not)
#   correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# 
#   # False is 0 and True is 1, what was our average?
#   accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# 
#   # How well did we do?
#   print(sess.run(accuracy, feed_dict={x: testFeatures, y_: testLabels}))
# 
#   # Close the session (self documenting code)
#   sess.close()
