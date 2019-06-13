from __future__ import absolute_import, division, print_function

import argparse
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from torchvision import transforms

from nets.lenet import LeNet
from master import Master
from worker import Worker, load_femnist_dataset
from datasets.femnist import FEMNISTDataset

DEFAULT_ARGS = {
    'epochs': 10,
    'batch_size': 32,
    'lr': 0.001,
    'log_every_n_steps': 10,
    'seed': 1234,
    'init_method': 'tcp://127.0.0.1:23456',
}


def run_master(device, model, args):
    # TODO: support multiple datasets
    test_dataset = FEMNISTDataset(args.dataset_dir, train=False,
                                  transform=transforms.ToTensor(),
                                  only_digits=True)
    if args.validation_period:
        client_ids = list(test_dataset.client_ids)
        if args.max_num_users:
            rng = random.Random()
            rng.seed(args.seed)
            rng.shuffle(client_ids)
            del client_ids[args.max_num_users:]

        test_datasets = [test_dataset.create_dataset(client_id)
                         for client_id in client_ids]

        test_data_loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset(test_datasets),
            batch_size=args.batch_size)

        test_args = test_data_loader, args.validation_period
    else:
        test_args = None

    master = Master(model, device, args.num_workers, args.init_method,
                    sample_size=args.sample_size, seed=args.seed)
    master.run(args.epochs, test_args)


def run_worker(rank, device, model, args):
    # TODO: support multiple datasets
    dataset = load_femnist_dataset(args.dataset_dir, rank, args.num_workers,
                                   args.seed, args.max_num_users,
                                   download=args.dataset_download)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True)

    worker = Worker(model, device, rank, args.num_workers, args.init_method,
                    sample_size=args.sample_size, seed=args.seed)
    worker.run(args.epochs, args.lr, data_loader, args.log_every_n_steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_workers', type=int, required=True,
        help='number of workers to use in simulation')
    parser.add_argument(
        '--sample_size', type=int,
        help='size of worker samples in each epoch. '
             'If not set, all workers paricipate in each epoch.')
    parser.add_argument(
        '--dataset_dir', required=True, help='dataset root dir')
    parser.add_argument(
        '--dataset_download', action='store_true',
        help='download the dataset if not exists')
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
    parser.add_argument(
        '--validation_period', type=int,
        help='period of validation that runs on the master')

    args = parser.parse_args()

    # TODO: support GPU
    device = torch.device('cpu')

    model = LeNet()

    master_process = mp.Process(target=run_master, args=(device, model, args))
    master_process.start()
    time.sleep(0.1)

    worker_processes = []
    for rank in range(1, args.num_workers + 1):
        p = mp.Process(target=run_worker,
                       args=(rank, device, model, args))
        p.start()
        worker_processes.append(p)

    master_process.join()
    for p in worker_processes:
        if p.is_alive():
            p.terminate()
        else:
            p.join()


if __name__ == "__main__":
    main()
