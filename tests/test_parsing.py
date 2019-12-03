"""Unit test for parsing methods"""
import unittest
import numpy as np
from dicom_data_preprocess import parsing
from dicom_data_preprocess.reader import DataReader
from dicom_data_preprocess.loader import DataLoader

__author__ = 'Christine Hsu'


class TestParsing(unittest.TestCase):

	def test_parse_contour_file_valid(self):
		print('\nTesting the parsing of a valid contour file...')

		valid_coords_lst = parsing._parse_contour_file(
			'tests/data/valid-icontour-manual.txt')
		self.assertTrue(isinstance(valid_coords_lst, list))
		self.assertTrue(isinstance(valid_coords_lst[0], tuple))
		self.assertTrue(isinstance(valid_coords_lst[0][0], float))
		self.assertTrue(len(valid_coords_lst) >= 3)

	def test_parse_contour_file_invalid(self):
		print('\nTesting the parsing of an invalid contour file with two coordinates...')

		self.assertRaises(Exception, parsing._parse_contour_file,
						  'tests/data/invalid-icontour-manual.txt')

	def test_parse_dicom_file_valid(self):
		print('\nTesting the parsing of a valid dicom file...')

		image = parsing._parse_dicom_file('tests/data/valid-dicom-file.dcm')
		self.assertTrue(isinstance(image, np.ndarray))

	def test_parse_dicom_file_invalid(self):
		print('\nTesting the parsing of an invalid dicom file with metadata removed')

		self.assertRaises(Exception, parsing._parse_dicom_file, 'tests/data/invalid-dicom-file.dcm')

	def test_poly_to_mask(self):
		print('\nTest the translation of a valid contour file to a boolean mask...')

		dcm_image = parsing._parse_dicom_file('tests/data/valid-dicom-file.dcm')
		coords_lst = parsing._parse_contour_file('tests/data/valid-icontour-manual.txt')
		mask = parsing._poly_to_mask(coords_lst, dcm_image.shape[1], dcm_image.shape[0])
		
		self.assertEqual(mask.shape, dcm_image.shape)
		self.assertEqual(mask.dtype, np.bool)


if __name__ == '__main__':
	unittest.main()
