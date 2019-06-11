from __future__ import absolute_import, division, print_function

import argparse
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision import transforms

from datasets.femnist import FEMNISTDataset, FEMNISTDatasetPartitioner
from nets.lenet import LeNet
from train import train_single_epoch


DEFAULT_ARGS = {
    'epochs': 10,
    'batch_size': 32,
    'lr': 0.001,
    'log_every_n_steps': 10,
    'seed': 1234,
    'init_method': 'tcp://127.0.0.1:23456',
}


def run_master(rank, device, model, args):
    dist.init_process_group('gloo', init_method=args.init_method,
                            rank=rank, world_size=args.num_devices + 1)

    for epoch in range(args.epochs):
        for parameter in model.parameters():
            dist.broadcast(parameter, 0)

        for parameter in model.parameters():
            parameter.data.zero_()
            dist.reduce(parameter, 0, op=dist.ReduceOp.SUM)
            parameter.data /= args.num_devices

        print('epoch={} finished'.format(epoch))


def run_worker(rank, device, model, args):
    dist.init_process_group('gloo', init_method=args.init_method,
                            rank=rank, world_size=args.num_devices + 1)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = F.nll_loss

    # TODO: support multiple datasets
    dataset = FEMNISTDataset(args.dataset_dir, download=False,
                             only_digits=True, transform=transforms.ToTensor())
    partitioner = FEMNISTDatasetPartitioner(
        dataset, args.num_devices, seed=args.seed)

    for epoch in range(args.epochs):
        log_prefix = 'rank={}, epoch={}'.format(rank, epoch)
        for parameter in model.parameters():
            dist.broadcast(parameter, 0)

        data_loader = torch.utils.data.DataLoader(
            partitioner.get(rank-1), batch_size=args.batch_size, shuffle=True)

        train_single_epoch(data_loader, model, optimizer, loss_fn,
                           args.log_every_n_steps, device, log_prefix)

        for parameter in model.parameters():
            dist.reduce(parameter, 0, op=dist.ReduceOp.SUM)


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
    parser.add_argument(
        '--seed', type=int, default=DEFAULT_ARGS['seed'],
        help='random seed (default={})'.format(DEFAULT_ARGS['seed']))
    parser.add_argument(
        '--init_method', default=DEFAULT_ARGS['init_method'],
        help='init method to use for torch.distributed (default={})'.format(
            DEFAULT_ARGS['init_method']))

    args = parser.parse_args()

    # TODO: support GPU
    device = torch.device('cpu')

    model = LeNet()

    processes = []
    for rank in range(args.num_devices + 1):
        if rank == 0:
            p = mp.Process(target=run_master,
                           args=(rank, device, model, args))
        else:
            p = mp.Process(target=run_worker,
                           args=(rank, device, model, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
