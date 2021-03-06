"""
Functions for managing DataSets of training, testing, and validation image data.

Inspired by:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/input_data.py
"""

import numpy
import os
from PIL import Image, ImageOps
import random
import tensorflow as tf


class DataSet(object):
    def __init__(self, images, labels, dtype=tf.float32):
        """
        Construct a DataSet.

        `dtype` can be either `uint8` to leave the input as `[0, 255]`,
        or `float32` to rescale into `[0, 1]`.
        """
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)
        assert images.shape[0] == labels.shape[0], (
                            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        assert images.shape[3] == 1

        # Store the width and height of the images before flattening it, if only for reference.
        image_height, image_width = images.shape[1], images.shape[2]
        self.original_image_width = image_width
        self.original_image_height = image_height

        images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
        if dtype == tf.float32:
            # Convert from [0, 255] -> [0.0, 1.0]
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
      """Return the next `batch_size` examples from this data set."""
      start = self._index_in_epoch
      self._index_in_epoch += batch_size
      if self._index_in_epoch > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1
          # Shuffle the data
          perm = numpy.arange(self._num_examples)
          numpy.random.shuffle(perm)
          self._images = self._images[perm]
          self._labels = self._labels[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size
          assert batch_size <= self._num_examples
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

def get_image_info_from_filename(image_filename):
    """
    Return num_edges, width, height from the given image_filename.

    image_filenames are like: polygon_4_480_640_145696672781244.png
    or full paths to a file with a basename of that form.
    """
    base_filename = os.path.basename(image_filename) # just the polygon_4_480_640_14....png part
    underscore_splits = base_filename.split('_')
    __, num_edges, width, height = underscore_splits[:4]
    num_edges, width, height = map(int, [num_edges, width, height])
    return num_edges, width, height

def is_valid_image_filename(image_filename):
    if not image_filename.endswith('.png'):
        return False
    return True

def get_image_filenames_of_size(width, height, images_dir):
    """
    Return a list of filenames in the given directory labelled with the given width and height.
    """
    filenames = []
    for filename in os.listdir(images_dir):
        if not is_valid_image_filename(filename):
            continue
        num_edges, img_width, img_height = get_image_info_from_filename(filename)
        if img_width == width and img_height == height:
            filenames.append(os.path.join(images_dir, filename))
    return filenames

def extract_images_and_labels(images_dir, width, height, num_labels=10):
    """
    Extracts from images in the given directory, of the given dimensions,
    into a 4D uint8 numpy array [index, y, x, depth],
    and a 2D numpy_array of one_hot labels of shape (num_images, num_labels)
    """
    eligible_files = get_image_filenames_of_size(width, height, images_dir=images_dir)
    num_images = len(eligible_files)

    # Initialize a numpy array to hold all the images, and reshape (only 3D for now)
    images_array = numpy.zeros(num_images * width * height, dtype=numpy.uint8)
    images_array = images_array.reshape(num_images, height, width)

    # Initalize a 1D array for the labels
    labels_array = numpy.zeros(num_images * num_labels, dtype=numpy.float32)
    labels_array = labels_array.reshape(num_images, num_labels)

    print("Extracting %s images and labels from %s/..." % (num_images, images_dir))

    # Shuffle the filenames to randomize the order of data.
    random.shuffle(eligible_files)

    for index, image_filename in enumerate(eligible_files):
        # Get info we'll need from the filename itself (we only really need the label)
        label, width_in_filename, height_in_filename = get_image_info_from_filename(image_filename)
        assert width_in_filename == width
        assert height_in_filename == height


        # double-withs is a workaround for PIL bug causing ResourceWarning: unclosed file
        # described here: https://github.com/python-pillow/Pillow/issues/835
        with open(image_filename, 'rb') as img_file:
            with Image.open(img_file) as open_pil_img:
                pil_image = open_pil_img.convert("L")
                pil_image = ImageOps.invert(pil_image)

        image_array = numpy.asarray(pil_image, dtype=numpy.uint8)
        images_array[index] = image_array

        # Now set the one_hot value for this label
        labels_array[index][label] = 1

    # Reshape to add a depth dimension
    images_array = images_array.reshape(num_images, width, height, 1)
    return images_array, labels_array

def read_data_sets(collection_directory, dtype=tf.float32):
    """
    Return a container DataSets object holding training and test DataSet objects extracted from
    images of the given size in /train and /test directiories under collection_dir.

    Requires all images be the same shape.
    """
    class DataSets(object):
        pass
    data_sets = DataSets()

    assert os.path.isdir(collection_directory)

    # Extact the image shape from the directory name, something like images/coll_28_28_1457065046/
    width, height = map(int, collection_directory.split('coll_')[1].split('_')[0:2])

    # Get our training and testing data sets
    train_dir = os.path.join(collection_directory, "train")
    test_dir = os.path.join(collection_directory, "test")

    train_images, train_labels = extract_images_and_labels(train_dir, width, height)
    test_images, test_labels = extract_images_and_labels(test_dir, width, height)

    data_sets.train = DataSet(train_images, train_labels, dtype=dtype)
    data_sets.test = DataSet(test_images, test_labels, dtype=dtype)

    return data_sets
