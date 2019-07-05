from __future__ import absolute_import, division, print_function

import argparse
import random
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

import datasets.femnist as femnist
from nets.lenet import LeNet
from centralized.master import Master
from utils import logger

_LOGGER = logger.get_logger(__file__, level=logger.DEBUG)


class Worker(object):

    def __init__(self, model, device, rank, num_workers, init_method,
                 backend='gloo', sample_size=None, seed=None):
        dist.init_process_group(backend, init_method=init_method, rank=rank,
                                world_size=(num_workers+1))

        self.model = model.to(device)
        self.device = device
        self.rank = rank
        self.num_workers = num_workers
        self.sample_size = sample_size

        if sample_size:
            assert seed, 'seed must be given in sampling target workers'
            self._rng = random.Random()
            self._rng.seed(seed)

    def _recv_params(self, group):
        for parameter in self.model.parameters():
            dist.broadcast(parameter, Master.RANK, group=group)

    def _reduce_params(self, group, op=dist.ReduceOp.SUM):
        for parameter in self.model.parameters():
            dist.reduce(parameter, Master.RANK, group=group, op=op)

    def _get_group(self):
        if not self.sample_size:
            return dist.group.WORLD, True

        targets = self._rng.sample(
            range(1, self.num_workers+1), self.sample_size)

        targets.append(Master.RANK)
        return dist.new_group(targets), (self.rank in targets)

    def run(self, epochs, local_epochs, train_args):
        for epoch in range(epochs):
            group, is_in_group = self._get_group()
            log_prefix = '[worker] rank: {}, epoch: [{}/{}]'.format(
                self.rank, epoch, epochs)
            self._recv_params(group)
            if is_in_group:
                for local_epoch in range(local_epochs):
                    new_log_prefix = '{}, local_epoch: [{}/{}]'.format(
                        log_prefix, local_epoch, local_epochs)
                    train_args.train_fn(train_args, new_log_prefix)
            self._reduce_params(group)


DEFAULT_ARGS = {
    'epochs': 10,
    'local_epochs': 10,
    'batch_size': 32,
    'lr': 0.001,
    'log_every_n_steps': 10,
    'seed': 1234,
    'init_method': 'tcp://127.0.0.1:23456',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_workers', type=int, required=True,
        help='number of workers')
    parser.add_argument(
        '--dataset_dir', required=True, help='dataset root dir')
    parser.add_argument(
        '--rank', type=int, required=True, help='rank')
    parser.add_argument(
        '--epochs', type=int, default=DEFAULT_ARGS['epochs'],
        help='number of epochs to train (default={})'.format(
            DEFAULT_ARGS['epochs']))
    parser.add_argument(
        '--local_epochs', type=int, default=DEFAULT_ARGS['local_epochs'],
        help='number of local epochs in each global epoch (default={})'.format(
            DEFAULT_ARGS['local_epochs']))
    parser.add_argument(
        '--batch_size', type=int, default=DEFAULT_ARGS['batch_size'],
        help='batch size (default={})'.format(DEFAULT_ARGS['batch_size']))
    parser.add_argument(
        '--lr', type=float, default=DEFAULT_ARGS['lr'],
        help='learning rate (default={})'.format(DEFAULT_ARGS['lr']))
    parser.add_argument(
        '--log_every_n_steps', type=int,
        default=DEFAULT_ARGS['log_every_n_steps'],
        help='log every n steps (default={})'.format(
            DEFAULT_ARGS['log_every_n_steps']))
    parser.add_argument(
        '--seed', type=int, default=DEFAULT_ARGS['seed'],
        help='random seed (default={})'.format(DEFAULT_ARGS['seed']))
    parser.add_argument(
        '--init_method', default=DEFAULT_ARGS['init_method'],
        help='init method to use for torch.distributed (default={})'.format(
            DEFAULT_ARGS['init_method']))
    parser.add_argument(
        '--max_num_users', type=int,
        help='max number of users that each worker can hold')

    args = parser.parse_args()

    if args.rank == Master.RANK:
        raise ValueError(
            'rank ({}) should be different from the master rank ({})'.format(
                args.rank, Master.RANK))

    dataset = femnist.get_partition(rank=args.rank-1,
                                    world_size=args.num_workers,
                                    train=True,
                                    only_digits=True,
                                    **vars(args))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True)

    model = LeNet()
    device = torch.device('cpu')

    worker = Worker(model, device, args.rank,
                    args.num_workers, args.init_method)
    worker.run(args.epochs, args.lr, args.local_epochs,
               data_loader, args.log_every_n_steps)


if __name__ == "__main__":
    main()
