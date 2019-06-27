from __future__ import absolute_import, division, print_function

import argparse
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

from datasets.femnist import FEMNISTDataset, FEMNISTDatasetPartitioner
from nets.lenet import LeNet
from decentralized.worker import Worker
from utils.train import TrainArguments


def run_worker(rank, args):
    dataset = FEMNISTDataset(args.dataset_dir, download=args.dataset_download,
                             only_digits=True, transform=transforms.ToTensor())
    partitioner = FEMNISTDatasetPartitioner(
        dataset,  args.num_workers, args.seed, args.max_num_users)
    partition = partitioner.get(rank)
    data_loader = torch.utils.data.DataLoader(
        partition, batch_size=args.batch_size, shuffle=True)

    validation = (None, None)
    if args.validation_period:
        test_dataset = FEMNISTDataset(args.dataset_dir, train=False,
                                      only_digits=True, transform=transforms.ToTensor())
        test_partitioner = FEMNISTDatasetPartitioner(
            test_dataset, args.num_workers, args.seed, args.max_num_users)
        test_partition = test_partitioner.get(rank)
        assert partition.client_ids == test_partition.client_ids
        test_data_loader = torch.utils.data.DataLoader(
            test_partition, batch_size=args.batch_size)
        validation = (args.validation_period, test_data_loader)

    admm_kwargs = None
    if args.use_admm:
        admm_kwargs = {
            'max_iter': args.admm_max_iter,
            'tolerance': args.admm_tolerance,
            'lr': args.admm_lr
        }

    model = LeNet()
    train_args = TrainArguments(
        data_loader=data_loader,
        device=torch.device('cpu'),
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=args.lr),
        loss_fn=F.nll_loss,
        log_every_n_steps=args.log_every_n_steps,
    )

    worker = Worker(rank, args.num_workers,
                    args.init_method, admm_kwargs=admm_kwargs)
    worker.run(args.epochs, args.local_epochs, train_args, validation)


DEFAULT_ARGS = {
    'epochs': 10,
    'local_epochs': 10,
    'batch_size': 32,
    'lr': 0.001,
    'log_every_n_steps': 10,
    'seed': 1234,
    'init_method': 'tcp://127.0.0.1:23456',
    'admm_tolerance': 0.01,
    'admm_max_iter': 20,
    'admm_lr': 0.01,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_workers', type=int, required=True,
        help='number of workers to use in simulation')
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
        '--use_admm', action='store_true',
        help='Use ADMM-based average for aggregation')
    parser.add_argument(
        '--admm_tolerance', type=float,
        default=DEFAULT_ARGS['admm_tolerance'],
        help='Tolerance for ADMM average')
    parser.add_argument(
        '--admm_max_iter', type=int,
        default=DEFAULT_ARGS['admm_max_iter'],
        help='max iteration for admm average')
    parser.add_argument(
        '--admm_lr', type=float,
        default=DEFAULT_ARGS['admm_lr'],
        help='learning rate for ADMM')
    parser.add_argument(
        '--validation_period', type=int,
        help='period of validation that runs on the master')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    processes = []
    for rank in range(args.num_workers):
        p = mp.Process(target=run_worker,
                       args=(rank, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
