"""
A simple shape classifier based on:
https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/tutorials/mnist/mnist_softmax.py#L6
"""

import argparse
import numpy
import os
import sys
import tempfile
import tensorflow as tf

from polygoggles import datasets
from polygoggles import make_polygon_pngs


def run_simple_softmax_from_scratch(image_width,
                                    image_height,
                                    num_train_images,
                                    num_test_images,
                                    num_training_steps=1000,
                                    training_batch_size=100,
                                    allow_rotations=False,
                                    save_images_to_dir=None):
    """
    Create a dataset according to the given parameters, then run a simple_softmax
    on the collection, and return the accuracy.
    """
    if not save_images_to_dir:
        with tempfile.TemporaryDirectory() as tmpdir:
            collection_dir = make_polygon_pngs.make_collection(image_width,
                                                               image_height,
                                                               num_train_images,
                                                               num_test_images,
                                                               root_dir=tmpdir,
                                                               allow_rotation=allow_rotations)
            accuracy = run_simple_softmax_on_collection(collection_dir, num_training_steps,
                                                        training_batch_size)
            print("Created then deleted:", collection_dir)
    else:
        collection_dir = make_polygon_pngs.make_collection(image_width,
                                                           image_height,
                                                           num_train_images,
                                                           num_test_images,
                                                           root_dir=save_images_to_dir,
                                                           allow_rotation=allow_rotations)
        accuracy = run_simple_softmax_on_collection(collection_dir, num_training_steps,
                                                    training_batch_size)
        print("Created directory:", collection_dir)
    return accuracy


def run_simple_softmax_on_collection(collection_directory, num_training_steps=1000,
                                     training_batch_size=100):
    """
    collection_directory should be the directory containing folders of /train and /test images,
    created via make_polygon_pngs or otherwise appropriately named with labels.
    """

    data_sets = datasets.read_data_sets(collection_directory)

    print("num train examples:", data_sets.train.num_examples)
    print("num test examples:", data_sets.test.num_examples)

    print("Shape of training image data:", data_sets.train.images.shape)
    print("Shape of training label data:", data_sets.train.labels.shape)

    # Make sure the data is in the form we're expecting.
    assert data_sets.train.images.ndim == 2
    assert data_sets.train.labels.ndim == 2

    assert data_sets.train.images.shape[1:] == data_sets.test.images.shape[1:]
    assert data_sets.train.labels.shape[1:] == data_sets.test.labels.shape[1:]

    print("Original image width:", data_sets.train.original_image_width)
    print("Original image height:", data_sets.train.original_image_height)

    edge_labels = data_sets.train.labels.shape[1] # number of different possible output labels

    image_flat_size = data_sets.train.images.shape[1]
    assert image_flat_size == data_sets.train.original_image_width * data_sets.train.original_image_height

    sess = tf.InteractiveSession()

    # Create the model
    x = tf.placeholder(tf.float32, [None, image_flat_size])
    W = tf.Variable(tf.zeros([image_flat_size, edge_labels]))
    b = tf.Variable(tf.zeros([edge_labels]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, edge_labels])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # Train
    tf.initialize_all_variables().run()
    for i in range(num_training_steps):
        batch_xs, batch_ys = data_sets.train.next_batch(training_batch_size)
        train_step.run({x: batch_xs, y_: batch_ys})

    # Test trained model and return accuracy on labeling test images
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_on_test_images = accuracy.eval({ x: data_sets.test.images, y_: data_sets.test.labels})
    print("Accuracy:", accuracy_on_test_images)
    return accuracy_on_test_images

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('collection_dir', help='Directory holding /train and /test')
    parser.add_argument('training_steps', default=1000, nargs='?', type=int,
                        help='Number of training steps (1000)')
    parser.add_argument('batch_size', default=100, nargs='?', type=int,
                        help='Number of images trained per batch.')
    args = parser.parse_args()
    run_simple_softmax_on_collection(args.collection_dir, args.training_steps, args.batch_size)


if __name__ == "__main__":
    main()
