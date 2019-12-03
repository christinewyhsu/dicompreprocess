"""Unit test for DataReader (public methods only)"""
import unittest
import numpy as np
import os
from dicom_data_preprocess import parsing
from dicom_data_preprocess.reader import DataReader

__author__ = 'Christine Hsu'


class TestReader(unittest.TestCase):
	@classmethod
	def setUpClass(TestReader):
		TestReader.download_data_path = 'tests/data/sample-batchset/'
		TestReader.data_basepath = 'tests/data/output_data/'
		TestReader.logs_path = 'tests/logs/',
		TestReader.plots_path = 'tests/plots/'
		TestReader.contour_type = 'i-contours'
		TestReader.save_plot = False

		TestReader.dicoms_basepath = os.path.join(TestReader.download_data_path, 'dicoms')
		TestReader.contours_basepath = os.path.join(TestReader.download_data_path, 'contourfiles')
		TestReader.link_filepath = os.path.join(TestReader.download_data_path, 'link.csv')

		link_tuples = DataReader._read_link(TestReader, TestReader.link_filepath)
		TestReader.sample_tuples = DataReader._assemble_link(TestReader, link_tuples)


	def test_load_samples(self):
		print('\nTesting the loading of eight assembled samples...')

		reader = DataReader(download_data_path=TestReader.download_data_path,
			data_basepath=TestReader.data_basepath,
			logs_path=TestReader.logs_path,
			plots_path=TestReader.plots_path,
			contour_type=TestReader.contour_type,
			save_plot=TestReader.save_plot)

		images, masks, metadata = reader.load_samples(TestReader.sample_tuples)
		self.assertTrue(isinstance(images, list))
		self.assertTrue(isinstance(masks, list))
		self.assertTrue(isinstance(metadata, list))
		self.assertTrue(isinstance(images[0], np.ndarray))
		self.assertEqual(masks[0].dtype, np.bool)
		self.assertTrue(isinstance(metadata[0], str))

		reader.plot_samples(images, masks, metadata, 'test_load_samples.jpg')


if __name__ == "__main__":
	unittest.main()
