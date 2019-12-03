"""Unit test for DataLoader (public methods only)"""
import unittest
import numpy as np
import os
from dicom_data_preprocess.loader import DataLoader

__author__ = 'Christine Hsu'


class TestLoader(unittest.TestCase):
	@classmethod
	def setUpClass(TestLoader):
		TestLoader.output_dir = 'tests/data/output_data/'
		TestLoader.images = np.load('tests/data/output_data/dicom_images.npy')
		TestLoader.masks = np.load('tests/data/output_data/mask_images.npy')
		TestLoader.metadata = np.load('tests/data/output_data/meta_images.npy')
		TestLoader.mini_batch_size = 2

	def test_random_mini_batches(self):
		print('Testing the shuffling and partitioning of the loader')
		loader = DataLoader(output_dir=TestLoader.output_dir,
			images=TestLoader.images,
			masks=TestLoader.masks,
			metadata=TestLoader.metadata,
			mini_batch_size=TestLoader.mini_batch_size)

		minibatches = loader.random_mini_batches()
		x, y, z = minibatches[0]
		self.assertEqual(len(minibatches), 4)
		self.assertEqual(np.shape(x)[0], 2)
		self.assertEqual(np.shape(y)[0], 2)
		self.assertEqual(np.shape(z)[1], 2)

if __name__ == '__main__':
	unittest.main()