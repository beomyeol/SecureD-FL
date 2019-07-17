from __future__ import absolute_import, division, print_function

import torch

from utils.test import test_model

class Worker(object):

    def __init__(self, rank, local_epochs, train_args, test_args):
        self.rank = rank
        self.local_epochs = local_epochs
        self.train_args = train_args
        self.test_args = test_args

    def train(self, log_prefix):
        for local_epoch in range(self.local_epochs):
            new_log_prefix = '{}, local_epoch: [{}/{}]'.format(
                log_prefix, local_epoch, self.local_epochs)
            self.train_args.train_fn(
                self.train_args, log_prefix=new_log_prefix)

    def test(self, log_prefix):
        test_model(self.test_args, log_prefix)

    @property
    def model(self):
        return self.train_args.model


def aggregate_models(workers, weights):
    with torch.no_grad():
        aggregated_state_dict = {}
        for worker, weight in zip(workers, weights):
            for name, parameter in worker.model.named_parameters():
                tensor = weight * parameter.data
                if name in aggregated_state_dict:
                    aggregated_state_dict[name] += tensor
                else:
                    aggregated_state_dict[name] = tensor

    return aggregated_state_dict
