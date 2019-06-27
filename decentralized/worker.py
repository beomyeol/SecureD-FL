from __future__ import absolute_import, division, print_function

import torch
import torch.distributed as dist

from utils import logger
from utils.train import train_single_epoch
from utils.test import test_model

_LOGGER = logger.get_logger(__file__)


class Worker(object):
    #TODO: support ADMM average

    def __init__(self, rank, num_workers, init_method,
                 backend='gloo', admm_kwargs=None):
        self.rank = rank
        self.num_workers = num_workers
        self.admm_kwargs = admm_kwargs

        dist.init_process_group(backend, init_method=init_method, rank=rank,
                                world_size=num_workers)

    def run(self, epochs, local_epochs, train_args, validation=(None, None)):
        validation_period, validation_loader = validation
        # CAVEATS: assume that model parameters of all workers are the same at the beginning.

        for epoch in range(epochs):
            log_prefix = '[worker] rank: {}, epoch: [{}/{}]'.format(
                self.rank, epoch, epochs)
            for local_epoch in range(local_epochs):
                new_log_prefix = '{}, local_epoch: [{}/{}]'.format(
                    log_prefix, local_epoch, local_epochs)
                train_single_epoch(train_args, log_prefix=new_log_prefix)

            self._average(train_args.model)

            if validation_period and epoch % validation_period == 0:
                test_model(validation_loader, train_args.model,
                           train_args.device, log_prefix)

    def _average(self, model):
        handles = []
        for parameter in model.parameters():
            handles.append(dist.all_reduce(
                parameter, op=dist.ReduceOp.SUM, async_op=True))

        with torch.no_grad():
            for i, parameter in enumerate(model.parameters()):
                handles[i].wait()
                parameter.data /= self.num_workers
