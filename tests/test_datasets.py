"""
Unittests for datasets.py
"""

import os
import unittest

from polygoggles.datasets import get_image_filenames_of_size, extract_images_and_labels

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGES_DIR = os.path.join(THIS_DIR, "images")
TEST_COLLECTION_DIR = os.path.join(TEST_IMAGES_DIR, "coll_28_28_1457066046")

class Test_get_image_filenames_of_size(unittest.TestCase):
    def test_gets_lone_42_by_23_filename(self):
        filenames = get_image_filenames_of_size(42, 23, images_dir=TEST_IMAGES_DIR)
        self.assertEqual(len(filenames), 1)
        filename_expected = os.path.join(TEST_IMAGES_DIR, "polygon_6_42_23_14569687645372.png")
        self.assertEqual(filenames[0], filename_expected)

class Test_read_data_sets(unittest.TestCase):
    def test_extract_images_and_labels(self):
        train_dir = os.path.join(TEST_COLLECTION_DIR, "train")
        images_4D_array, labels = extract_images_and_labels(train_dir, 28, 28, one_hot=True)

        # Make sure these numpy arrays are the right shape
        expected_images_array_shape = (8, 28, 28, 1)
        expected_labels_shape = (8,)
        self.assertEqual(images_4D_array.shape, expected_images_array_shape)
        self.assertEqual(labels.shape, expected_labels_shape)
        # Make sure it has the correct label
        self.assertEqual(labels[0], 3)


if __name__ == "__main__":
    unittest.main()
