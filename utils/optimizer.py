"""Optimizers."""
from __future__ import absolute_import, division, print_function

import torch.optim as optim


def get_optimizer(name, params, lr):
    if name == 'adam':
        return optim.Adam(params, lr=lr)
    elif name == 'rmsprop':
        return optim.RMSprop(params, lr=lr)
    else:
        raise ValueError('Unknown optimizer')
