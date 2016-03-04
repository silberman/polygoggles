"""
Unittests for make_polygon_pngs.py
"""

import os
import shutil
import unittest

from polygoggles.make_polygon_pngs import make_collection

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_BASE_IMAGES_DIR = os.path.join(THIS_DIR, "images")


class Test_make_collection(unittest.TestCase):

    def setUp(self):
        self.directories_to_tearDown = []

    def tearDown(self):
        for path in self.directories_to_tearDown:
            shutil.rmtree(path)

    def test_make_small_collection(self):
        collection_path = make_collection(5, 5, num_train_images=2, num_test_images=1,
                                          root_dir=TEST_BASE_IMAGES_DIR)
        self.directories_to_tearDown.append(collection_path)
        self.assertTrue(os.path.isdir(collection_path))
        train_dir = os.path.join(collection_path, "train")
        self.assertEqual(len(os.listdir(train_dir)), 2)

if __name__ == "__main__":
    unittest.main()
