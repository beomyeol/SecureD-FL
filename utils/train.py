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
        'train_fn',
        'writer',
        'test_fn',
    ])


def train_model(args, log_prefix='', rank=None, global_step=None):
    losses = []
    args.model.train()
    writer = args.writer

    running_loss = 0.0
    # for traing accuracy
    correct_sum = 0
    total_sum = 0

    for batch_idx, (data, target) in enumerate(args.data_loader):
        data, target = data.to(args.device), target.to(args.device)
        args.optimizer.zero_grad()
        pred = args.model(data)
        loss = args.loss_fn(pred, target)
        losses.append(loss.item())
        running_loss += loss.item()
        loss.backward()
        args.optimizer.step()
        if batch_idx % args.log_every_n_steps == 0:
            _LOGGER.info(
                log_prefix + (', ' if log_prefix else '') +
                'batches: [%d/%d], loss: %f',
                batch_idx, len(args.data_loader), loss.item())
        if writer is not None and batch_idx % 10 == 9:
            name = 'training loss'
            if rank is not None:
                name += '/worker#%d' % rank
            step = None if global_step is None else global_step + batch_idx
            writer.add_scalar(name, running_loss / 10, step)
            running_loss = 0.0
        if args.test_fn is not None:
            with torch.no_grad():
                correct, total = args.test_fn(pred, target)
            correct_sum += correct
            total_sum += total

    if writer is not None and args.test_fn is not None:
        name = 'training accuracy'
        if rank is not None:
            name += '/worker#%d' % rank
        step = global_step + len(args.data_loader)
        writer.add_scalar(name, correct_sum/total_sum, step)

    return losses


def train_rnn(args, hidden, log_prefix=''):
    losses = []
    hidden = hidden.to(args.device)
    args.model.train()
    for batch_idx, (data, target) in enumerate(args.data_loader):
        data, target = data.to(args.device), target.to(args.device)
        hidden = hidden.detach()
        args.optimizer.zero_grad()
        out, hidden = args.model(data, hidden)
        loss = args.loss_fn(out, target)
        losses.append(loss.item())
        loss.backward()
        args.optimizer.step()
        if batch_idx % args.log_every_n_steps == 0:
            _LOGGER.info(
                log_prefix + (', ' if log_prefix else '') +
                'batches: [%d/%d], loss: %f',
                batch_idx, len(args.data_loader), loss.item())
    return losses
