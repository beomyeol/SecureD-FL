from __future__ import absolute_import, division, print_function

import argparse
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.hub import tqdm

from nets.lenet import LeNet


class Master(object):

    RANK = 0

    def __init__(self, model, device, num_workers, init_method, backend='gloo',
                 seed=None):
        dist.init_process_group(backend, init_method=init_method,
                                rank=self.RANK, world_size=(num_workers+1))

        self.model = model.to(device)
        self.device = device
        self.num_workers = num_workers

    def _broadcast_params(self):
        for parameter in self.model.parameters():
            dist.broadcast(parameter, self.RANK)

    def _clear_params(self):
        for parameter in self.model.parameters():
            parameter.data.zero_()

    def _average_params(self):
        for parameter in self.model.parameters():
            dist.reduce(parameter, self.RANK, op=dist.ReduceOp.SUM)
            parameter.data /= self.num_workers

    def run_validation(self, data_loader):
        loss = 0
        num_correct = 0
        total_num = 0

        def loss_fn(input, target):
            return F.nll_loss(input, target, reduction='sum')

        self.model.eval()

        print('Run Validation...')
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

    def run(self, epochs, test_data_loader=None):
        for epoch in range(epochs):
            self._broadcast_params()
            # Previous parameter values should be cleared.
            # Otherwise, they will be included in the reduced values.
            self._clear_params()
            self._average_params()

            log = '[master] epoch: [{}/{}]'.format(epoch, epochs)

            if test_data_loader:
                loss, accuracy = self.run_validation(test_data_loader)
                log += ', test_loss: {}, test_accuracy: {}'.format(
                    loss, accuracy)

            print(log)


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
