"""
Source code of preprocessor
"""
import argparse
from reader import DataReader
from loader import DataLoader
import json
import numpy as np
import logging
import os
import sys


__author__ = 'Christine Hsu'


def _setup_logging(logs_path):

	LOG_FILENAME = os.path.join(logs_path, 'preprocessor.log')
	logging.root.setLevel(logging.NOTSET)
	logging.basicConfig(filename=LOG_FILENAME,level=logging.NOTSET)

	
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.DEBUG)

def main(params):
	# Arguments passed down from the parser
	download_data_path = params['input_data_path']
	data_basepath = params['output_data_path']
	logs_path = params['logs_path']
	plots_path = params['plots_path']
	contour_type = params['contour_type']
	toggle_plot = params['toggle_plot']
	mini_batch_size = params['mini_batch_size']

	# Set up logging
	_setup_logging(logs_path)

	# Meat of the python program
	logging.info('Started running preprocessor for the following parameters: {}'.format(params))
	reader = DataReader(download_data_path=download_data_path, data_basepath=data_basepath,
						logs_path=logs_path, plots_path=plots_path, contour_type=contour_type, save_plot=toggle_plot)
	images, masks, metadata = reader.load_samples(reader.sample_tuples)
	loader = DataLoader(output_dir=data_basepath, images=images,
						masks=masks, metadata=metadata, mini_batch_size=mini_batch_size)
	minibatches = loader.random_mini_batches()

	# If user enabled the toggle_plot to evaluate the reader and loader modules
	if toggle_plot:
		# Check out the overall view of all samples (dicoms, masks) with no shuffle and no partitioning
		logging.debug('Plotting the overall view of all (dicom, mask) samples...')
		reader.plot_samples(images, masks, metadata,
							'data-reader_no-shuffle_batchset.jpg')

		# Check out first minibatch to see whether it matches the ones in 'data-reader_no-shuffle_batchset.jpg' with same label
		logging.debug(
			'Extracting and plotting the first minibatch to validate DataLoader against the previous plot from DataReader...')
		for i, minibatch in enumerate(minibatches):
			if i > 1:
				break
			minibatch_image, minibatch_mask, minibatch_metadata = minibatch

		# minibatch_image (8,256,256), minibatch_mask (8,256,256), minibatch_metadata (8,)
		reader.plot_samples(minibatch_image, minibatch_mask,
							minibatch_metadata, 'data-loader_shuffled_batchset.jpg')
		logging.info('Finished running preprocessor...')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Set up input and output paths given the target contour type of medical imaging analysis')

	parser.add_argument('--input_dir', dest='input_data_path', type=str,
						default='data/final_data',  help='dataset: download_data')
	parser.add_argument('--output_dir', dest='output_data_path', type=str, default='data/output_data',
						help='output directory to cache parsed images and masks and csv file of sample tuples')
	parser.add_argument('--logs_dir',  dest='logs_path',       type=str, default='logs',
						help='output directory to write the log files to')
	parser.add_argument('--plots_path', dest='plots_path',       type=str, default='plots',
						help='output directory to save the figure with subplots for visual inspection')
	parser.add_argument('--contour_type', dest='contour_type', type=str, default='i-contours',
						choices=['i-contours', 'o-contours'], help='The contour type under investigation')
	parser.add_argument('--toggle_plot', dest='toggle_plot',   default=True, action='store_true',
						help='Generate overview plot of all overlay subplot per each pair of DICOM image and mask file')
	parser.add_argument('--mini_batch_size', dest='mini_batch_size',
						type=int, default=8, help='Size of mini batch')

	args = parser.parse_args()
	params = vars(args)
	print('parsed parameters: ')
	print(json.dumps(params, indent=2))

	main(params)
