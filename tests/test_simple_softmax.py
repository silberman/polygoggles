"""
Unittests for simple_softmax.py
"""

import os
import random
import shutil
import tempfile
import unittest

from polygoggles.simple_softmax import run_simple_softmax_from_scratch

class Test_softmax(unittest.TestCase):

    def setUp(self):
        self.temp_collection_dir = tempfile.mkdtemp()
        print("temp dir:", self.temp_collection_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_collection_dir)

    def test_28x28_norotation(self):
        width, height = 28, 28
        accuracy = run_simple_softmax_from_scratch(width, height, 1000, 50, allow_rotations=False)
        self.assertGreater(accuracy, 0.0)

    def test_saving_images_from_scratch(self):
        # Test the path through run_simple_softmax_from_scratch that doesn't use tempfile
        accuracy = run_simple_softmax_from_scratch(28, 28, 5, 2, 2, 1,
                                                   save_images_to_dir=self.temp_collection_dir)
        self.assertEqual(1, len(os.listdir(self.temp_collection_dir))) # a coll_ directory


if __name__ == "__main__":
    unittest.main()
