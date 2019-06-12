from __future__ import absolute_import, division, print_function

import argparse
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

from datasets.femnist import FEMNISTDataset, FEMNISTDatasetPartitioner
from nets.lenet import LeNet
from train import train_single_epoch
from master import Master


class Worker(object):

    def __init__(self, model, device, rank, num_workers, init_method,
                 backend='gloo'):
        dist.init_process_group(backend, init_method=init_method, rank=rank,
                                world_size=(num_workers+1))

        self.model = model.to(device)
        self.device = device
        self.rank = rank
        self.num_workers = num_workers

    def _recv_params(self):
        for parameter in self.model.parameters():
            dist.broadcast(parameter, Master.RANK)

    def _reduce_params(self, op=dist.ReduceOp.SUM):
        for parameter in self.model.parameters():
            dist.reduce(parameter, Master.RANK, op=op)

    def run(self, epochs, lr, data_loader, log_every_n_steps):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = F.nll_loss

        for epoch in range(epochs):
            log_prefix = '[worker] rank: {}, epoch: [{}/{}]'.format(
                self.rank, epoch, epochs)
            self._recv_params()
            train_single_epoch(data_loader, self.model, optimizer, loss_fn,
                               log_every_n_steps, self.device, log_prefix)
            self._reduce_params()


DEFAULT_ARGS = {
    'epochs': 10,
    'batch_size': 32,
    'lr': 0.001,
    'log_every_n_steps': 10,
    'seed': 1234,
    'init_method': 'tcp://127.0.0.1:23456',
}


def load_femnist_dataset(root_dir, rank, num_workers, seed, max_num_users,
                         download=True):
    dataset = FEMNISTDataset(root_dir, download=download,
                             only_digits=True, transform=transforms.ToTensor())
    partitioner = FEMNISTDatasetPartitioner(
        dataset, num_workers, seed, max_num_users)

    return partitioner.get(rank-1)


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
        '--batch_size', type=int, default=DEFAULT_ARGS['batch_size'],
        help='batch size (default={})'.format(DEFAULT_ARGS['batch_size']))
    parser.add_argument(
        '--lr', type=float, default=DEFAULT_ARGS['lr'],
        help='learning rate (default={})'.format(DEFAULT_ARGS['lr']))
    parser.add_argument(
        '--log_every_n_steps',
        type=int,
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

    dataset = load_femnist_dataset(
        args.dataset_dir, args.rank, args.num_workers, args.seed,
        args.max_num_users)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True)

    model = LeNet()
    device = torch.device('cpu')

    worker = Worker(model, device, args.rank,
                    args.num_workers, args.init_method)
    worker.run(args.epochs, args.lr, data_loader, args.log_every_n_steps)


if __name__ == "__main__":
    main()
