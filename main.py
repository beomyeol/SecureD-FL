from __future__ import absolute_import, division, print_function

import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

from datasets.femnist import FEMNISTDataset, FEMNISTDatasetPartitioner
from nets.lenet import LeNet


DEFAULT_ARGS = {
    'epochs': 10,
    'batch_size': 32,
    'lr': 0.001,
    'log_every_n_steps': 10,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_devices', type=int, required=True,
        help='num devices to use in simulation')
    parser.add_argument(
        '--dataset_dir', required=True, help='dataset root dir')
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

    args = parser.parse_args()

    # TODO: support GPU
    device = torch.device('cpu')

    # TODO: support multiple datasets
    dataset = FEMNISTDataset(args.dataset_dir, download=True,
                             only_digits=True, transform=transforms.ToTensor())
    partitioner = FEMNISTDatasetPartitioner(dataset, args.num_devices)

    data_loaders = []
    for device_idx in range(args.num_devices):
        data_loaders.append(torch.utils.data.DataLoader(
            partitioner.get(device_idx),
            batch_size=args.batch_size,
            shuffle=True))

    model = LeNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for device_idx, data_loader in enumerate(data_loaders):
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                pred = model(data)
                loss = F.nll_loss(pred, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('epoch: {}, devices: [{}/{}], '
                          'batches: [{}/{}], loss: {}'.format(
                              epoch, device_idx, len(data_loaders),
                              batch_idx, len(data_loader), loss.item()))


if __name__ == "__main__":
    main()
