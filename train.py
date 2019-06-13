from __future__ import absolute_import, division, print_function

import torch

from utils import logger

_LOGGER = logger.get_logger(__file__)


def train_single_epoch(data_loader,
                       model,
                       optimizer,
                       loss_fn,
                       log_every_n_steps,
                       device,
                       log_prefix):
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_every_n_steps == 0:
            _LOGGER.info(log_prefix + ', batches: [%d/%d], loss: %f',
                         batch_idx, len(data_loader), loss.item())
