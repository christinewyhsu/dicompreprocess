"""
DataLoader is responsible for shuffling the inputs (DICOM images) and labels (masks)
and partitioning the single batch into minibatches of mini_batch_size.

"""
import os
import numpy as np
import math
import logging


__author__ = 'Christine Hsu'
__all__ = ['DataLoader']

logger = logging.getLogger(__name__)


class DataLoader(object):

    def __init__(self, output_dir, images, masks, metadata, mini_batch_size):
        """
        Args:
            output_dir (str): Path to save the minibatches
            images (array): array of parsed DICOM images (X)
            masks (array):  array of parsed contour labels translated into boolean masks (Y)
            metadata (array): array of metadata (Y)
        """
        self.output_dir = output_dir
        self.images = np.asarray(images)
        self.masks = np.asarray(masks)
        self.metadata = np.asarray(metadata)
        self.mini_batch_size = mini_batch_size

        # Determine the number of samples
        self.m_samples = self.images.shape[0]
        # Determine the number of features
        self.num_features = self.images.shape[1]

    def random_mini_batches(self):
        """
        This method builds mini-batches from the (dicom images, contour labels).

        Args:
            mini_batch_size (int): the desired number of examples per batch

        Return:
            An list of mini-batches
        """
        # Set up empty array for partitioning the shuffled samples into mini-batches of size mini_batch_size
        mini_batches = []
        logging.debug('Shuffling the {} samples'.format(self.m_samples))

        # Create a shuffled version of the samples
        permutation = list(np.random.permutation(self.m_samples))
        shuffled_images = self.images[permutation, :, :]
        shuffled_masks = self.masks[permutation, :, :]
        # Adds another dimension to the metadata array
        sample_id = np.expand_dims(self.metadata, axis=0)
        shuffled_metadata = sample_id[:, permutation]

        # Partition the shuffled version of the samples into mini-batches of size mini_batch_size
        num_complete_minibatches = math.floor(
            self.m_samples / self.mini_batch_size)
        for k in range(0, num_complete_minibatches):
            logging.debug('Partitioning the {}-minibatch of {} samples'.format(k+1, self.mini_batch_size))

            mini_batch_images = shuffled_images[k *self.mini_batch_size: (k+1)*self.mini_batch_size, :, :]
            mini_batch_masks = shuffled_masks[k *self.mini_batch_size: (k+1)*self.mini_batch_size, :, :]
            mini_batch_metadata = shuffled_metadata[:, k *self.mini_batch_size: (k+1)*self.mini_batch_size]

            mini_batch = (mini_batch_images, mini_batch_masks,mini_batch_metadata)
            mini_batches.append(mini_batch)

        # In case when the number of training examples is not divisible by mini_batch_size
        if (self.m_samples % self.mini_batch_size) != 0:
            logging.debug('Partitioning the remaining minibatch of {} samples'.format(
                self.m_samples % self.mini_batch_size))

            mini_batch_images = shuffled_images[num_complete_minibatches *self.mini_batch_size:, :, :]
            mini_batch_masks = shuffled_masks[num_complete_minibatches *self.mini_batch_size:, :, :]
            mini_batch_metadata = shuffled_metadata[:,num_complete_minibatches*self.mini_batch_size:]

            mini_batch = (mini_batch_images, mini_batch_masks,mini_batch_metadata)
            mini_batches.append(mini_batch)
        
        x, y, z = mini_batches[0]
        print(np.shape(x))
        print(np.shape(y))
        print(np.shape(z))
        logging.debug('Finished shuffling and partitioning the single batch into {} minibatches. Now saving shuffled_minibatch_dataset.npy'.format(
            len(mini_batches)))
        mini_batch_path = os.path.join(self.output_dir, 'shuffled_minibatch_dataset.npy')
        np.save(mini_batch_path, mini_batches)

        return mini_batches
