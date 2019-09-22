from __future__ import absolute_import, division, print_function

from datasets import femnist
from datasets import shakespeare


def get_load_dataset_fn(name):
    name = name.lower()
    if name == 'femnist':
        return femnist.load_dataset
    elif name == 'shakespeare':
        return shakespeare.load_dataset
    else:
        raise ValueError('Unknown dataset: ' + name)
