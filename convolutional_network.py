import tensorflow as tf
# import cigar_input as ci
import plant_input as ci
import numpy as np
import sys, random
import os
import argparse

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

# Two functions to create weights and biases. Slightly positive bias due to
# using ReLU neurons. This is all to avoid noise for symmetry breaking
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    initial = tf.Variable(initial)
    return initial

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    initial = tf.Variable(initial)
    return initial

# Our convolutions use a stride of one and are zero padded so that the output
# is the same size as the input. Using 2x2 blocks for pooling
def conv2d(x, W):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return conv

def max_pool_2x2(x):
    pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
    return pool


def parse_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train','--trainDir', type=str)
    parser.add_argument('-test','--testDir', type=str)
    parser.add_argument('-e','--numEpochs', type=int)
    parser.add_argument('-bs','--trainBatchSize', type=int)
    parser.add_argument('-c1','--numConvs1', type=int)
    parser.add_argument('-c2','--numConvs2', type=int)

    args = parser.parse_args()
    return args


def run():
    args = parse_user_args()

    trainDir = args.trainDir
    testDir = args.testDir
    numEpochs = args.numEpochs
    trainBatchSize = args.trainBatchSize
    numConvs1 = args.numConvs1
    numConvs2 = args.numConvs2

    numLabels = 4
    inputPixels = 128
    numColorChannels = 1

    # number of convolutions applied to single input image in first layer
    numFilters1 = numConvs1 * 1

    # number of convolutions applied to the resulting filters of the
    # first convolutional layer
    numFilters2 = numConvs2 * numFilters1

    # number of pixels of our convolution patches for first and second layers
    patchSize1 = 5
    patchSize2 = 5
    
    # post-convolution pooling sizes
    poolSize1 = 2
    poolSize2 = 2

    # since we're doing padding around the edges our final reduced image is only
    # affected by the size of our poolings. If we were not padding feature maps
    # before convolving, we would have to subtract the number of pixels lost in
    # convolution from the input, and then divide by pooling size
    reducedImagePixels =  int(inputPixels/poolSize1/poolSize2)

    # Placeholders for our input images and labels
    x = tf.placeholder("float", shape=[None, inputPixels**2])
    y_ = tf.placeholder("float", shape=[None, numLabels])


    ###
    ### FIRST CONVOLUTION LAYER
    ###

    # Reshape image to 4d tensor. Second and third dimensions are image width
    # and height. Final dimension corresponds to number of color channels. 
    # -1 in size calculates what shape should be to have other values constant
    x_image = tf.reshape(x, [-1,
                             inputPixels,
                             inputPixels,
                             numColorChannels])

    # The first two dimensions are patch size, the next is the number of input
    # channels, and the last is the number of output channels.
    W_conv1 = weight_variable([patchSize1,
                               patchSize1,
                               numColorChannels,
                               numFilters1])

    b_conv1 = bias_variable([numFilters1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # First pooling
    h_pool1 = max_pool_2x2(h_conv1)


    ###
    ### SECOND CONVOLUTION LAYER
    ###

    W_conv2 = weight_variable([patchSize2,
                               patchSize2,
                               numFilters1,
                               numFilters2])

    b_conv2 = bias_variable([numFilters2])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # Second pooling
    h_pool2 = max_pool_2x2(h_conv2)


    ###
    ### FULLY CONNECTED LAYER
    ###

    W_fc1 = weight_variable([reducedImagePixels**2 * numFilters2,
                             numFilters1**2])
    b_fc1 = bias_variable([numFilters1**2])
    # We reshape the tensor from the pooling layer into a batch of vectors,
    # multiply by a weight matrix, add a bias, and apply a ReLU
    h_pool2_flat = tf.reshape(h_pool2, [-1,reducedImagePixels**2 * numFilters2])
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
    ### DEFINE OBJECTIVE FUNCTION, OPTIMIZER, and ACCURACY
    ###

    # added a very small value for numerical stability
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv + 1e-9))
    # We use the ADAM optimizer instead of steepest gradient descent
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  

    ###
    ### TRAINING
    ###
  
    output = open('output.txt', 'w')

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    for subDir in os.listdir(trainDir):
        dataPath = os.path.join(trainDir,subDir)
        print(dataPath)
        trainX,trainY = ci.input_data(dataPath)

        for i in range(numEpochs):
            batch_xs, batch_ys = get_random_subset(trainBatchSize,trainX,trainY)

            if i%10 == 0:
                # use keep_prob in feed_dict to control dropout rate
                train_accuracy = accuracy.eval(feed_dict={x: batch_xs,
                                                          y_: batch_ys,
                                                          keep_prob: 1.0})

                print("step %d, training accuracy %g"%(i, train_accuracy))

                for subDir in os.listdir(testDir):
                    dataPath = os.path.join(testDir,subDir)
                    testX,testY = ci.input_data(dataPath)
                    test_accuracy = accuracy.eval(feed_dict={x: testX, 
                                                             y_: testY, 
                                                             keep_prob: 1.0})
                    print((str(i) +'\t'+ 
                           str(train_accuracy) +'\t'+ 
                           str(test_accuracy)),
                          file=output)

            train_step.run(feed_dict={x: batch_xs, 
                                      y_: batch_ys, 
                                      keep_prob: 0.5})
if __name__ == "__main__":
    run()
