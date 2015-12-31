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
  # print(features[tempIndexes,:])
  # print(labels[tempIndexes,:])
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

# Two functions to create weights and biases. Slightly positive bias due to
# using ReLU neurons. This is all to avoid noise for symmetry breaking
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Our convolutions use a stride of one and are zero padded so that the output
# is the same size as the input. Using 2x2 blocks for pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')



sess = tf.InteractiveSession()

path = sys.argv[1]

numEpochs = 5
trainBatchSize = 128
numLabels = 4
inputPixels = 128
numColorChannels = 1
numFilters1 = 10
numFilters2 = 20
convPixels1 = 5
convPixels2 = 5
poolSize1 = 2
poolSize2 = 2
reducedImagePixels = int(inputPixels/poolSize1/poolSize2)

x = tf.placeholder("float", shape=[None, inputPixels**2])
y_ = tf.placeholder("float", shape=[None, numLabels])

###
### FIRST CONVOLUTION LAYER
###

# The first two dimensions are the patch size, the next is the number of input
# channels, and the last is the number of output channels.
W_conv1 = weight_variable([convPixels1,convPixels1,numColorChannels,
                           numFilters1])
b_conv1 = bias_variable([numFilters1])
# Reshape image to 4d tensor. The second and third dimensions are image width
# and height. Final dimension corresponds to number of color channels. 
# -1 in size calculates what the shape should be to have other values constant
x_image = tf.reshape(x, [-1,inputPixels,inputPixels,numColorChannels])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# First pooling
h_pool1 = max_pool_2x2(h_conv1)

###
### SECOND CONVOLUTION LAYER
###

W_conv2 = weight_variable([convPixels2, convPixels2, numFilters1, numFilters2])
b_conv2 = bias_variable([numFilters2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# Second pooling
h_pool2 = max_pool_2x2(h_conv2)

###
### FULLY CONNECTED LAYER
###

W_fc1 = weight_variable([reducedImagePixels**2 * numFilters2, numFilters1**2])
b_fc1 = bias_variable([numFilters1**2])
# We reshape the tensor from the pooling layer into a batch of vectors,
# multiply by a weight matrix, add a bias, and apply a ReLU
h_pool2_flat = tf.reshape(h_pool2, [-1, reducedImagePixels**2 * numFilters2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

###
### APPLY DROPOUT
###

# Placeholder holds probability of a neuron getting kept
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

###
### FINAL SOFTMAX LAYER
###
W_fc2 = weight_variable([numFilters1**2, numLabels])
b_fc2 = bias_variable([numLabels])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

###
### TRAINING
###

# added a very small value for numerical stability
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv + 1e-9))
# We use the ADAM optimizer instead of steepest gradient descent
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

trainFeatures,trainLabels,testFeatures,testLabels = get_test_train_sets(path)

sess.run(tf.initialize_all_variables())
for i in range(numEpochs):
  batch_xs, batch_ys = get_random_subset(trainBatchSize,trainFeatures,
                                           trainLabels)
  if i%1 == 0:
    # use keep_prob in feed_dict to control dropout rate
    train_accuracy = accuracy.eval(feed_dict={
      x:batch_xs, y_: batch_ys, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

print('Results of test run after training')  
print(sess.run(accuracy, feed_dict={x: testFeatures, y_: testLabels, keep_prob: 1.0}))
