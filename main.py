from __future__ import absolute_import, division, print_function

import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

from datasets.femnist import FEMNISTDataset
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
    loaders = {}
    for client_id in dataset.client_ids:
        loaders[client_id] = torch.utils.data.DataLoader(
            dataset.create_dataset(client_id),
            batch_size=args.batch_size,
            shuffle=True)

    model = LeNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for c_idx, (c_id, c_loader) in enumerate(loaders.items()):
            for batch_idx, (data, target) in enumerate(c_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                pred = model(data)
                loss = F.nll_loss(pred, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('epoch: {}, clients: [{}/{}], '
                          'batches: [{}/{}], loss: {}'.format(
                              epoch, c_idx, len(loaders),
                              batch_idx, len(c_loader), loss.item()))


if __name__ == "__main__":
    main()
