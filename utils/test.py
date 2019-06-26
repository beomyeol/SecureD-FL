from __future__ import absolute_import, division, print_function

import torch

from utils import logger

_LOGGER = logger.get_logger(__file__)


def test_model(data_loader, model, device, log_prefix):
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output.data, dim=1)
            total += target.size(0)
            correct += (pred == target).sum().item()

    _LOGGER.info(log_prefix + ', test accuracy: %s', str(correct/total))
