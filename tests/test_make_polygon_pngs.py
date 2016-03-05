"""
Unittests for make_polygon_pngs.py
"""

import os
import shutil
import tempfile
import unittest

from polygoggles.make_polygon_pngs import make_collection


class Test_make_collection(unittest.TestCase):

    def setUp(self):
        self.temp_collection_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_collection_dir)

    def test_make_small_collection(self):
        collection_path = make_collection(5, 5, num_train_images=2, num_test_images=1,
                                          root_dir=self.temp_collection_dir)

        self.assertTrue(os.path.isdir(collection_path))
        train_dir = os.path.join(collection_path, "train")
        self.assertEqual(len(os.listdir(train_dir)), 2)

    def test_makes_the_right_amount(self):
        collection_path = make_collection(5, 5, num_train_images=20000, num_test_images=1,
                                          root_dir=self.temp_collection_dir)
        train_dir = os.path.join(collection_path, "train")
        self.assertEqual(20000, len(os.listdir(train_dir)))

if __name__ == "__main__":
    unittest.main()
