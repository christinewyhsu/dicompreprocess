"""
DataReader is responsible for parsing the raw DICOM images and contour files from repo-dir/download_data,
translating contour files into boolean mask of the contours, and saving the parsed DICOM image, boolean mask
of the contours, and metadata in repo-dir/dicom_data_preprocess/data/. It also contains a plot toggle to enable
generating and saving overview of all overlay plots per each pair of DICOM image and contour/mask file for visual inspection
in repo-dir/dicom_data_preprocess/plots/.

DICOM images without matching annotations are skipped. Moreover, if contour_type ('i-contour')
is inputted, only DICOM images with corresponding i-contour labels are parsed.
"""
import os
from os import path
import csv
import glob
import logging
import parsing
import matplotlib.pyplot as plt
import numpy as np


__author__ = 'Christine Hsu'
__all__ = ['DataReader']

logger = logging.getLogger(__name__)


class DataReader(object):

    class InvalidContourFilenameError(Exception):
        """Exception raised when a contour filename is unrecognizable"""
        pass

    def __init__(self, download_data_path,
                 data_basepath,
                 logs_path,
                 plots_path,
                 contour_type,  # Only focusing on one type of contour files at a time
                 save_plot):
        """
        Args:
            download_data_path (str):    Path to the directory where the raw DICOM images and contour files is downloaded
            data_basepath (str):         Basepath of the directory where the images and masks numpy arrays are saved
            logs_path (str):             Path of the directory where the log files are saved
            plots_path (str):            Path of the directory where the plots are saved 
            contour_type (enum):         'i-contours', 'o-contours'
            save_plot (boolean):         Enables whether to generate and save plots in the <plots_path>
        """

        # Set variables to the arguments passed in

        self.download_data_path = download_data_path
        self.data_basepath = data_basepath
        self.logs_path = logs_path
        self.plots_path = plots_path
        self.contour_type = contour_type
        self.save_plot = save_plot

        # Condition when user enter contour type without 's'
        if self.contour_type in ('i-contour', 'o-contour'):
            self.contour_type = self.contour_type + 's'

        # Log if user enters unrecognizable contour type
        if self.contour_type not in ['i-contours', 'o-contours']:
            logging.warning(
                'Entered invalid contour type: ' + self.contour_type)

        # Set path variables to dicoms directory, contourfiles directory, link.csv in download_data folder
        self.dicoms_basepath = os.path.join(download_data_path, 'dicoms')
        self.contours_basepath = os.path.join(
            download_data_path, 'contourfiles')
        self.link_filepath = os.path.join(download_data_path, 'link.csv')

        link_tuples = self._read_link(self.link_filepath)
        self.sample_tuples = self._assemble_link(link_tuples)

    def _read_link(self, filename):
        """
        A method that reads the input csv file that links up the appropriate dicoms and contour files

        Args:
            filename (str): csv filename

        Return:
            list of tuples (dicom dir, contour dir)
        """

        logging.info('Reading link.csv file...')

        link_tuples = []

        with open(filename) as csvFile:
            reader = csv.DictReader(csvFile)
            for row in reader:
                link_tuples.append((row['patient_id'], row['original_id']))

        logging.debug(link_tuples)

        return link_tuples

    def _assemble_link(self, link_tuples):
        """
        A method that iterates over (dicom dir, contour dir) and searches for all contour files and
        matching DICOM file. If 'i-contours' or 'o-contours' is given in constructor as contour_type,
        then only DICOM files with i-contour or o-contour files will be found, respectively.

        The method also creates a csv file output of the list of tuples (contour filepath, DICOM filepath)
        for debugging purposes.

        Args:
           link_tuples (list): list of tuples (dicom dir, contour dir)

        Return:
           sample_tuples (list): list of tuples (contour filepath, DICOM filepath)
           Csv file of the sample_tuples for debuggin purposes in the repo-dir/dicom_data_preprocess/
        """

        sample_tuples = []


        # Iterate over the list of tuples (dicom dir, contour dir)
        for dicom_dir, contour_dir in link_tuples:

            logging.info('Querying available dicome files in {} corresponding to {} files in {}...'.format(
                dicom_dir, self.contour_type, contour_dir))

            # Extract all contour filepaths given contour type ('i-contour','o-contour')
            # Define file format before inputting into path.join()
            fileformat = self.contour_type + '/*.txt'
            contours_paths = glob.glob(
                path.join(self.contours_basepath, contour_dir, fileformat))
            print(contours_paths)

            # Find matching DICOM file for contour file
            for contour_file in sorted(contours_paths):

                # Split contour filename for sample identifier set-up
                parts = path.basename(contour_file).split('-')
                if parts[0] != "IM" or parts[1] != '0001':
                    logging.warning(
                        'Unknown contour filenaming pattern: ' + contour_file)
                    raise DataReader.InvalidContourFilenameError('''Contour filename pattern is unrecognizable: {}'''.format(contour_file))

                # Extract sample identifier from contour filename
                sample_id = parts[2].lstrip('0')
                sample_id = sample_id + '.dcm'

                # Create a temp dicom filename for dicom file search query
                dicom_file = os.path.join(
                    self.dicoms_basepath, dicom_dir, sample_id)

                # If dicom file is found, then create tuple of (contour file, dicom file)
                if path.isfile(dicom_file):
                    sample_tuples.append((contour_file, dicom_file))
                    logging.debug('{} is now linked with {}'.format(
                        contour_file, dicom_file))
                else:
                    logging.warning('Missing dicom for contour: ' + dicom_path)

        # Output csv file that specifies the contour filepath and its corresponding dicom filepath (if any)
        # into repo-dir/dicom_data_preprocess/

        logging.debug('Writing sample tuples into csv file...')

        filepath = os.path.join(self.data_basepath, 'sample_pairs.csv')
        with open(filepath, 'w') as outfile:
            mywriter = csv.writer(outfile)
            mywriter.writerow(['contour_filename', 'dicom_filename'])
            for row in sample_tuples:
                mywriter.writerow(row)

        return sample_tuples

    def load_samples(self, sample_tuples):
        """
        This method calls the other methods from the 'parsing.py' module to
        parse the DICOM file, parse the contour file and translate it to boolean mask, 
        as well as save the DICOM file and boolean mask to repo-dir/dicom-data-preprocess/data
        in numpy array format. 

        Caching the DICOM file and boolean mask is to validate the parsing methods, which
        can be done through visual inspection

        Args:
           sample_tuples (tuple): list of tuples (contour filename, dicom filename)

        Return:
           A single batch of all DICOM images and its corresponding mask labels
        """

        images = []
        masks = []
        metadata = []

        for contour_file, dicom_file in sample_tuples:

            # Parse contour and dicome filenames for logging and saving filename purposes
            # patient_id extracted from dicom filename as in link.csv
            patient_id = dicom_file.split('/')[-2]
            # original_id extracted from contour filename as in link.csv
            original_id = contour_file.split('/')[-3]
            dcm_filename = dicom_file.split('/')[-1]
            contour_filename = contour_file.split('/')[-1]
            # sample identifier that links the dicom and contour files
            sample_id = contour_filename.split('-')[2].lstrip('0')

            identifiers = [patient_id, original_id, 'sample-id', sample_id]
            meta = "_".join(identifiers)

            logging.debug('patient_id({}), original_id({}): Loading dicom images and masks with sample id {}'.format(
                patient_id, original_id, sample_id))

            # Parse the single DICOM file and single contour file, as well as translate contour
            # file to mask
            dcm_img = parsing._parse_dicom_file(dicom_file)
            coords_lst = parsing._parse_contour_file(contour_file)
            # shape[1]-Width of Image, shape[0]-Height of Image
            mask = parsing._poly_to_mask(
                coords_lst, dcm_img.shape[1], dcm_img.shape[0])

            images.append(dcm_img)
            masks.append(mask)
            metadata.append(meta)

        # Save the single batch of dicom image, mask, metadata numpy arrays
        logging.debug('Finished loading all samples and now saving dicom_images.npy, mask_images.npy, and meta_images.npy to {}'.format(
            self.data_basepath))
        img_path = os.path.join(self.data_basepath, 'dicom_images.npy')
        mask_path = os.path.join(self.data_basepath, 'mask_images.npy')
        metadata_path = os.path.join(self.data_basepath, 'meta_images.npy')
        np.save(img_path, images)
        np.save(mask_path, masks)
        np.save(metadata_path, metadata)

        return images, masks, metadata

    def plot_samples(self, dicoms, masks, metadata, filename):
        """
        This method plots the DICOM images with the available mask label

        Args:
            dicoms (array): A 3-D array that contains shape of (number of samples, width of image, height of image)
            masks (array):  A 3-D array that contains shape of (number of samples, width of image, height of image)
            metadata (array): An array that contains shape of (number of samples,)
            filename (str): The name of the jpg file that the user would like name plot
        Return:
            An overall view of all images for visual inspection
        """
        # Determine the number of samples for dicoms, masks
        dicom_m = np.shape(dicoms)[0]
        mask_m = np.shape(masks)[0]
        metadata_m = np.shape(metadata)[0]

        if dicom_m != mask_m != metadata_m:
            logging.warning('''The number of mask labels ({}), the number of dicom image examples ({}), 
                and the number of metadata identifiers do not match'''.format(mask_m, dicom_m, metadata_m))

        # Settings for subplot
        h, w = 256, 256
        ncols = 4
        nrows = int(
            dicom_m/ncols) if dicom_m % ncols == 0 else int(dicom_m/ncols) + 1
        print(nrows)
        if nrows <= 2:
            figsize = [20, 10]
            fontsize = 10
        else:
            figsize = [50, 250]
            fontsize = 18

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        # Iterate over all the ax[row_id][col_id]
        for i, ax in enumerate(axes.flat):

            # If the number of samples are uneven, then plot dummy images; otherwise just
            # plot the overlay of the dicom image and mask along with the metadata
            if i > (dicom_m-1):
                img = np.random.randint(10, size=(h, w))
                mask = np.random.randint(10, size=(h, w))
                title = ''
            else:
                img = dicoms[i]
                mask = masks[i]
                metadata = np.squeeze(metadata)
                title = metadata[i]

            ax.imshow(img, cmap='gray', aspect='auto')
            ax.imshow(mask, alpha=0.25, aspect='auto')
            ax.set_title(title, fontsize=fontsize)
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        plt.tight_layout()

        # Save the overall view of the no shuffle batchset from data_reader
        filepath = os.path.join(self.plots_path, filename)
        fig.savefig(filepath)
