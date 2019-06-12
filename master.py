from __future__ import absolute_import, division, print_function

import argparse
import torch
import torch.distributed as dist

from nets.lenet import LeNet


class Master(object):

    def __init__(self, model, device, rank, num_workers,
                 init_method, backend='gloo'):
        dist.init_process_group(backend, init_method=init_method, rank=rank,
                                world_size=(num_workers+1))

        self.model = model.to(device)
        self.device = device
        self.rank = rank
        self.num_workers = num_workers

    def _broadcast_params(self):
        for parameter in self.model.parameters():
            dist.broadcast(parameter, self.rank)

    def _clear_params(self):
        for parameter in self.model.parameters():
            parameter.data.zero_()

    def _average_params(self):
        for parameter in self.model.parameters():
            dist.reduce(parameter, self.rank, op=dist.ReduceOp.SUM)
            parameter.data /= self.num_workers

    def run(self, epochs):
        for epoch in range(epochs):
            self._broadcast_params()
            # Previous parameter values should be cleared.
            # Otherwise, they will be included in the reduced values.
            self._clear_params()
            self._average_params()

            print('[master] epoch: [{}/{}]'.format(epoch, epochs))


DEFAULT_ARGS = {
    'epochs': 10,
    'rank': 0,
    'init_method': 'tcp://127.0.0.1:23456',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_workers', type=int, required=True,
        help='number of workers')
    parser.add_argument(
        '--rank', type=int, default=DEFAULT_ARGS['rank'],
        help='rank (default={})'.format(DEFAULT_ARGS['rank']))
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

    master = Master(model, device, args.rank,
                    args.num_workers, args.init_method)
    master.run(args.epochs)


if __name__ == "__main__":
    main()
