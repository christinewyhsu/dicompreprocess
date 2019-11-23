'''BatchLoader is essentially the refactored code of Pytorch data_loading_tutorial.html.

'''

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import linecache

class DicomMasksDataset(Dataset):
    ''' Dicom & mask images dataset.'''

    def __init__(self, pair_path, transform=None):
        '''
        Create a dataset object to iterate over a file containing in each line a path to:
        img, contour-1-mask, contour-2-mask, etc... numpy arrays.
        Parameters
        ----------
        pair_path: basestring
            Path of the file where are saved image,masks paths per line.
        transform: callable or None
            A transformation to apply to a training sample.
            Each sample being: a dict(image, contour-1-mask, contour-2-mask).
            An example of  function can be a conversion from numpy array to Pytorch/Keras/Tensorflow/... tensors.
        '''

        samples_count = 0
        with open(pair_path, 'rb') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                # first line for the header:
                if idx == 0:
                    self.headers = line.split(' ')
                    continue
                # next lines contains paths to img, and contours-masks paths
                if line:
                    samples_count += 1

        self.samples_count = samples_count
        self.pair_path = pair_path
        self.transform = transform

    def __len__(self):
        return self.samples_count

    def __getitem__(self, idx):
        """
        Return transformed sample
        """

        # Add 2 since linecache starts index at 1 (not 0)
        idx += 2

        paths = linecache.getline(self.pair_path, idx).strip().split(' ')
        sample_paths = dict(zip(self.headers, paths))
        sample = dict((key, np.load(path)) for key, path in sample_paths.iteritems())
        if self.transform:
            sample = self.transform(sample)

        return sample

class DataLoaderProcessor(object):
    @staticmethod
    def get_data_loader(pair_path, tensor_or_ndarray='ndarray', batch_size=8, shuffle=True, num_workers=1):
        '''
        Encasulate the creation of the Pytorch Dataset object and then the DataLoader object given the parameters.

        :param pair_path: basestring
            Path of the file where are saved a pair of image,mask paths per line.
        :param tensor_or_ndarray: basestring
           Decides to return a torch tensor or a numpy ndarray: values to choose between ndarray and tensor
        :param batch_size: int
            The number of samples per batch
        :param shuffle: bool
            Load the samples randomly from the entire dataset.
        :param num_workers: int
            If multi-core/GPUs use this parameter to load data in parallel.

        Returns
        -------
        data_loader: DataLoader
             which generates either Torch Tensors or numpy ndarrays per batch
             depending of the value of tensor_or_ndarray.
        '''

        # first instantiate the dataset object
        dataset = DicomMasksDataset(pair_path=pair_path, transform=convert_to_tensor)

        # now create the data_loader object
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        if tensor_or_ndarray == 'ndarray':
            data_loader = (convert_to_ndarray(sample) for sample in data_loader)

        return data_loader


def convert_to_tensor(sample):
    '''
    Transform a ndarray array sample into a Tensorflow sample

    :param sample: dict ('img':np.int16, 'i-contour':np.bool, 'o-contour':np.bool, ...)

    Returns
    -------
    sample: dict ('img':Tensorflow.ShortTensor,
                  'i-contour':Tensorflow.ByteTensor,
                  'o-contour':Tensorflow.ByteTensor,
                  ...)
    '''
    new_sample = dict()
    for key, val in sample.iteritems():
        if key == 'img':
            new_sample[key] = torch.from_numpy(val)
        else: # handle contour mask differently
            new_sample[key] = torch.from_numpy(val.astype(np.uint8))

    return new_sample

def convert_to_ndarray(sample):
    '''
    Transform a Tensorflow array sample into a ndarray sample

    :param sample: dict ('img':Tensorflow.ShortTensor,
                  'i-contour':Tensorflow.ByteTensor,
                  'o-contour':Tensorflow.ByteTensor,
                  ...)

    Returns
    -------
    sample: dict ('img':np.int16, 'i-contour':np.bool, 'o-contour':np.bool, ...)
    '''
    new_sample = dict()
    for key, val in sample.iteritems():
        if key == 'img':
            new_sample[key] = val.numpy()
        else:  # handle contour mask differently
            new_sample[key] = val.numpy().astype(np.bool)

    return new_sample