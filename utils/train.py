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
        'test_fn',
    ])


def train_model(args, log_prefix=''):
    loss_sum = 0.0
    args.model.train()

    running_loss = 0.0
    # for train accuracy
    correct_count = 0

    for batch_idx, (data, target) in enumerate(args.data_loader):
        data, target = data.to(args.device), target.to(args.device)
        args.optimizer.zero_grad()
        pred = args.model(data)
        loss = args.loss_fn(pred, target)
        loss_sum += loss.item() * len(pred)
        running_loss += loss.item()
        loss.backward()
        args.optimizer.step()
        if batch_idx % args.log_every_n_steps == 0:
            log_fmt = log_prefix
            if log_prefix:
                log_fmt += ', '
            log_fmt += 'batches: [%d/%d], loss: %f'
            _LOGGER.info(log_fmt,
                         batch_idx, len(args.data_loader),
                         running_loss / (batch_idx + 1))

        if args.test_fn is not None:
            with torch.no_grad():
                correct, total = args.test_fn(pred, target)
            correct_count += correct

    metrics = {'count': len(args.data_loader.dataset), 'loss_sum': loss_sum}

    if args.test_fn is not None:
        metrics['correct_count'] = correct_count

    return metrics


def train_rnn(args, hidden, log_prefix=''):
    loss_sum = 0.0
    hidden = hidden.to(args.device)
    args.model.train()

    running_loss = 0.0
    # for train accuracy
    correct_count = 0
    total_count = 0  # count number of characters, not sentences.

    for batch_idx, (data, target) in enumerate(args.data_loader):
        data, target = data.to(args.device), target.to(args.device)
        hidden = hidden.detach()
        args.optimizer.zero_grad()
        out, hidden = args.model(data, hidden)
        loss = args.loss_fn(out, target)
        loss_sum += loss.item() * len(out)
        running_loss += loss.item()
        loss.backward()
        args.optimizer.step()
        if batch_idx % args.log_every_n_steps == 0:
            log_fmt = log_prefix
            if log_prefix:
                log_fmt += ', '
            log_fmt += 'batches: [%d/%d], loss: %f'
            _LOGGER.info(log_fmt,
                    batch_idx, len(args.data_loader),
                         running_loss / (batch_idx + 1))

        if args.test_fn is not None:
            with torch.no_grad():
                correct, total = args.test_fn(out, target)
            correct_count += correct
            total_count += total

    metrics = {
        'count': total_count,
        'loss_sum': loss_sum
    }

    if args.test_fn is not None:
        metrics['correct_count'] = correct_count

    return metrics
