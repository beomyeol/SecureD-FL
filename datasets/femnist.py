from __future__ import absolute_import, division, print_function

import h5py
import os.path
import torch.utils.data
from torchvision import transforms

from datasets.utils import download_and_extract_archive
import datasets.partition as partition
import utils.logger as logger


_LOGGER = logger.get_logger(__file__)


class FEMNISTClientDataset(torch.utils.data.Dataset):

    def __init__(self, client_h5, transform=None, target_transform=None):
        self._images = client_h5['pixels']
        self._labels = client_h5['label']
        self._transform = transform
        self._target_transform = target_transform

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        image = self._images[index, :]
        label = self._labels[index].item()

        if self._transform:
            image = self._transform(image)

        if self._target_transform:
            label = self._target_transform(label)

        return image, label


class FEMNISTDataset(object):

    _EXAMPLE_GROUP = 'examples'
    _BASE_URL = 'https://storage.googleapis.com/tff-datasets-public/'

    def __init__(self, root, train=True, download=False, transform=None,
                 target_transform=None, only_digits=False,
                 max_num_clients=None):
        self._root = root
        # Try to use the dataset generated from the LEAF dataset first
        self._fileprefix = 'femnist'
        if only_digits:
            self._fileprefix += '_digitsonly'

        if not self._check_exists():
            # If not exists, use TensorFlow's dataset
            self._fileprefix = self._fileprefix.replace(
                'femnist', 'fed_emnist')

        if download:
            self.download()

        if train:
            data_file = self.train_file
        else:
            data_file = self.test_file

        _LOGGER.info('Loading FEMNIST at %s', data_file)

        self._h5_file = h5py.File(data_file, 'r')
        self._transform = transform
        self._target_transform = target_transform
        self.client_ids = sorted(list(self._h5_file['examples'].keys()))
        if max_num_clients is not None:
            del self.client_ids[max_num_clients:]

    @property
    def train_file(self):
        return os.path.join(self._root, self._fileprefix + '_train.h5')

    @property
    def test_file(self):
        return os.path.join(self._root, self._fileprefix + '_test.h5')

    def _check_exists(self):
        return (os.path.exists(self.train_file) and
                os.path.exists(self.test_file))

    def download(self):
        if self._check_exists():
            return

        download_and_extract_archive(
            self._BASE_URL + self._fileprefix + '.tar.bz2',
            download_root=self._root,
            remove_finished=True
        )

    def create_dataset(self, client_id):
        client_h5 = self._h5_file[self._EXAMPLE_GROUP][client_id]
        return FEMNISTClientDataset(client_h5,
                                    transform=self._transform,
                                    target_transform=self._target_transform)


def load_dataset(dataset_dir, train=True, dataset_download=False,
                 only_digits=True, max_num_clients=None, **kwargs):
    return FEMNISTDataset(dataset_dir,
                          train=train,
                          download=dataset_download,
                          only_digits=only_digits,
                          transform=transforms.ToTensor(),
                          max_num_clients=max_num_clients)
