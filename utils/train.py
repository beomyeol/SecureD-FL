from __future__ import absolute_import, division, print_function

import collections
import time
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
        'train_fn',
    ])


def train_model(args, log_prefix=''):
    args.model.train()
    t = time.time()
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
    _LOGGER.info(log_prefix + (', ' if log_prefix else '') +
                 'time: %s sec', str(time.time() - t))


def train_rnn(args, hidden, log_prefix=''):
    args.model.train()
    t = time.time()
    for batch_idx, (data, target) in enumerate(args.data_loader):
        data, target = data.to(args.device), target.to(args.device)
        hidden = hidden.detach()
        args.optimizer.zero_grad()
        out, hidden = args.model(data, hidden)
        loss = args.loss_fn(out, target)
        loss.backward()
        args.optimizer.step()
        if batch_idx % args.log_every_n_steps == 0:
            _LOGGER.info(
                log_prefix + (', ' if log_prefix else '') +
                'batches: [%d/%d], loss: %f',
                batch_idx, len(args.data_loader), loss.item())
    _LOGGER.info(log_prefix + (', ' if log_prefix else '') +
                 'time: %s sec', str(time.time() - t))
