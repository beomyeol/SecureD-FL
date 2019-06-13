from __future__ import absolute_import, division, print_function

import argparse
import random
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.hub import tqdm

from nets.lenet import LeNet
from utils import logger

_LOGGER = logger.get_logger(__file__, level=logger.DEBUG)


class Master(object):

    RANK = 0

    def __init__(self, model, device, num_workers, init_method, backend='gloo',
                 sample_size=None, seed=None):
        dist.init_process_group(backend, init_method=init_method,
                                rank=self.RANK, world_size=(num_workers+1))

        self.model = model.to(device)
        self.device = device
        self.num_workers = num_workers
        self.sample_size = sample_size

        if sample_size:
            assert seed, 'seed must be given in sampling target workers'
            self._rng = random.Random()
            self._rng.seed(seed)

    def _get_group(self):
        if not self.sample_size:
            return dist.group.WORLD

        targets = self._rng.sample(
            range(1, self.num_workers+1), self.sample_size)
        targets.append(self.RANK)
        _LOGGER.debug('[master] targets: %s', str(targets))
        return dist.new_group(targets)

    def _broadcast_params(self, group):
        for parameter in self.model.parameters():
            dist.broadcast(parameter, self.RANK, group=group)

    def _clear_params(self):
        for parameter in self.model.parameters():
            parameter.data.zero_()

    def _average_params(self, group):
        for parameter in self.model.parameters():
            dist.reduce(parameter, self.RANK, group=group, op=dist.ReduceOp.SUM)
            parameter.data /= self.num_workers

    def run_validation(self, data_loader):
        loss = 0
        num_correct = 0
        total_num = 0

        def loss_fn(input, target):
            return F.nll_loss(input, target, reduction='sum')

        self.model.eval()

        _LOGGER.info('Run Validation...')
        with torch.no_grad():
            pbar = tqdm(total=len(data_loader))
            for data, target in data_loader:
                out = self.model(data)
                loss += loss_fn(out, target)
                preds = torch.argmax(out, dim=1)
                num_correct += (preds == target).sum()
                total_num += len(target)
                pbar.update(1)

            loss /= total_num
            accuracy = num_correct.float() / total_num

        return loss, accuracy

    def run(self, epochs, test_args=None):
        for epoch in range(epochs):
            group = self._get_group()
            self._broadcast_params(group)
            # Previous parameter values should be cleared.
            # Otherwise, they will be included in the reduced values.
            self._clear_params()
            self._average_params(group)

            log = '[master] epoch: [{}/{}]'.format(epoch, epochs)

            if test_args:
                test_data_loader, test_period = test_args
                if epoch % test_period == 0:
                    loss, accuracy = self.run_validation(test_data_loader)
                    log += ', test_loss: {}, test_accuracy: {}'.format(
                        loss, accuracy)

            _LOGGER.info(log)


DEFAULT_ARGS = {
    'epochs': 10,
    'init_method': 'tcp://127.0.0.1:23456',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_workers', type=int, required=True,
        help='number of workers')
    parser.add_argument(
        '--epochs', type=int, default=DEFAULT_ARGS['epochs'],
        help='number of epochs to train (default={})'.format(
            DEFAULT_ARGS['epochs']))
    parser.add_argument(
        '--init_method', default=DEFAULT_ARGS['init_method'],
        help='init method to use for torch.distributed (default={})'.format(
            DEFAULT_ARGS['init_method']))

    args = parser.parse_args()

    model = LeNet()
    device = torch.device('cpu')

    master = Master(model, device, args.num_workers, args.init_method)
    master.run(args.epochs)


if __name__ == "__main__":
    main()
