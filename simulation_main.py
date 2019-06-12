from __future__ import absolute_import, division, print_function

import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nets.lenet import LeNet
from master import Master
from worker import Worker, load_femnist_dataset


DEFAULT_ARGS = {
    'epochs': 10,
    'batch_size': 32,
    'lr': 0.001,
    'log_every_n_steps': 10,
    'seed': 1234,
    'init_method': 'tcp://127.0.0.1:23456',
}


def run_master(rank, device, model, args):
    master = Master(model, device, rank, args.num_workers, args.init_method)
    master.run(args.epochs)


def run_worker(rank, master_rank, device, model, args):
    # TODO: support multiple datasets
    dataset = load_femnist_dataset(args.dataset_dir, rank, args.num_workers,
                                   args.seed, args.max_num_users)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True)

    worker = Worker(model, device, rank, master_rank, args.num_workers,
                    args.init_method)
    worker.run(args.epochs, args.lr, data_loader, args.log_every_n_steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_workers', type=int, required=True,
        help='number of workers to use in simulation')
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
    parser.add_argument(
        '--max_num_users', type=int,
        help='max number of users that each worker can hold')

    args = parser.parse_args()

    # TODO: support GPU
    device = torch.device('cpu')

    model = LeNet()

    processes = []
    master_rank = 0
    for rank in range(args.num_workers + 1):
        if rank == master_rank:
            p = mp.Process(target=run_master,
                           args=(rank, device, model, args))
        else:
            p = mp.Process(target=run_worker,
                           args=(rank, master_rank, device, model, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
