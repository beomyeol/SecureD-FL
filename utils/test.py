from __future__ import absolute_import, division, print_function

import torch

import collections
from utils import logger

_LOGGER = logger.get_logger(__file__)


TestArguments = collections.namedtuple(
    'TestArguments',
    [
        'data_loader',
        'model',
        'device',
        'period',
        'test_fn',
        'writer',
    ]
)


def test_model(args, log_prefix, rank=None, global_step=None):
    correct_sum = 0
    total_sum = 0
    args.model.eval()

    with torch.no_grad():
        for data, target in args.data_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = args.model(data)
            correct, total = args.test_fn(output, target)
            correct_sum += correct
            total_sum += total

    _LOGGER.info(log_prefix + ', test accuracy: %s[%d/%d]',
                 str(correct_sum/total_sum), correct_sum, total_sum)

    if args.writer is not None:
        name = 'validation accuracy'
        if rank is not None:
            name += '/worker#%d' % rank
        args.writer.add_scalar(name, correct_sum/total_sum, global_step)
