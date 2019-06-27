from __future__ import absolute_import, division, print_function

import collections
import torch

from utils import logger

_LOGGER = logger.get_logger(__file__)


TrainArguments = collections.namedtuple(
    'TrainArguments',
    [
        'data_loader',
        'device',
        'model',
        'optimizer',
        'loss_fn',
        'log_every_n_steps',
    ])


def train_single_epoch(args, log_prefix=''):
    args.model.train()
    for batch_idx, (data, target) in enumerate(args.data_loader):
        data, target = data.to(args.device), target.to(args.device)
        args.optimizer.zero_grad()
        pred = args.model(data)
        loss = args.loss_fn(pred, target)
        loss.backward()
        args.optimizer.step()
        if batch_idx % args.log_every_n_steps == 0:
            _LOGGER.info(
                log_prefix + (', ' if log_prefix else '') +
                'batches: [%d/%d], loss: %f',
                batch_idx, len(args.data_loader), loss.item())
