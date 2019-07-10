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
    ]
)


def test_model(args, log_prefix):
    correct = 0
    total = 0
    args.model.eval()

    with torch.no_grad():
        for data, target in args.data_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = args.model(data)
            if type(output) == tuple:
                output = output[0]
            _, pred = torch.max(output.data, dim=1)
            if len(target.shape) > 1:
                total += target.size(1)
            else:
                total += target.size(0)
            correct += (pred == target).sum().item()

    _LOGGER.info(log_prefix + ', test accuracy: %s[%d/%d]',
                 str(correct/total), correct, total)
