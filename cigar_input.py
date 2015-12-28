import tensorflow.python.platform
import tensorflow as tf
import os

def read_cigar(dir):
  """Reads and parses examples from our cigar data files.
  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.
  Args:
    dir: Directory to read in the files
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (128)
      width: number of columns in the result (128)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """
  class CigarRecord(object):
    pass
  result = CigarRecord()
  # Dimensions of the images in the dataset.
  label_bytes = 1  # We only have 4 labels
  result.height = 128
  result.width = 128
  result.depth = 3
  image_bytes = result.height * result.width * result.depth
  # Get all files to read from current directory
  filenames = os.listdir(dir)
  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes
  # Read a record, getting filenames from the filename_queue.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)
  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)
  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                           [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])
  return result