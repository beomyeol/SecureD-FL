from __future__ import absolute_import, division, print_function

import h5py
import os.path
import torch.utils.data

from datasets.utils import download_and_extract_archive
import datasets.partition as partition
import utils.logger as logger


_LOGGER = logger.get_logger(__file__)
VOCAB = list(
    'dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r}')


class ShakespeareClientDataset(torch.utils.data.Dataset):

    def __init__(self, client_h5, client_id, transform=None):
        self._chunks = client_h5['snippets']

        if transform:
            transformed_list = []
            for chunk in self._chunks:
                transformed_list.extend(transform(chunk))
            self._chunks = transformed_list

    def __len__(self):
        return len(self._chunks)

    def __getitem__(self, index):
        return self._chunks[index]


class ShakespeareDataset(object):

    _EXAMPLE_GROUP = 'examples'
    _URL = 'https://storage.googleapis.com/tff-datasets-public/shakespeare.tar.bz2'
    CHAR2IDX = {u: i for i, u in enumerate(VOCAB)}
    IDX2CHAR = VOCAB

    def __init__(self, root, train=True, download=False, transform=None):
        self._root = root

        if download:
            self.download()

        if train:
            data_file = self.train_file
        else:
            data_file = self.test_file

        self._h5_file = h5py.File(data_file, 'r')
        self._transform = transform
        self.client_ids = sorted(list(self._h5_file['examples'].keys()))

    @property
    def train_file(self):
        return os.path.join(self._root, 'shakespeare_train.h5')

    @property
    def test_file(self):
        return os.path.join(self._root, 'shakespeare_test.h5')

    def _check_exists(self):
        return (os.path.exists(self.train_file) and
                os.path.exists(self.test_file))

    def download(self):
        if self._check_exists():
            return

        download_and_extract_archive(
            self._URL,
            download_root=self._root,
            remove_finished=False)

    def create_dataset(self, client_id):
        client_h5 = self._h5_file[self._EXAMPLE_GROUP][client_id]
        return ShakespeareClientDataset(client_h5, client_id,
                                        transform=self._transform)


def batching(text, seq_length):
    batches = [text[i:i+seq_length]
               for i in range(0, len(text), seq_length)]
    if batches and len(batches[-1]) < seq_length:
        del batches[-1]
    return batches


def split_input_target(chunk):
    x = chunk[:-1]
    target = chunk[1:]
    return x, target


class Preprocessor(object):

    def __init__(self,
                 seq_length,
                 char2idx=ShakespeareDataset.CHAR2IDX):
        self.seq_length = seq_length
        self.char2idx = char2idx

    def __call__(self, text):
        text = text.decode(encoding='utf-8')
        encoded = [self.char2idx[c] for c in text]
        chunks = []
        for chunk in batching(encoded, self.seq_length+1):
            if not chunk:
                pass
            x, target = split_input_target(chunk)
            x = torch.LongTensor(x)
            target = torch.LongTensor(target)
            chunks.append((x, target))
        return chunks


def load_dataset(dataset_dir, seq_length, train=True, dataset_download=False,
                 **kwargs):
    preprocessor = Preprocessor(seq_length)
    return ShakespeareDataset(dataset_dir,
                              train=train,
                              download=dataset_download,
                              transform=preprocessor)