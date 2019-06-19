from __future__ import absolute_import, division, print_function

import argparse
import torch
import torch.multiprocessing as mp
from torchvision import transforms
import uuid

from datasets.femnist import FEMNISTDataset, FEMNISTDatasetPartitioner
from leader_based.worker import Worker
from leader_based.zk_election import ZkElection
from nets.lenet import LeNet


def run_worker(rank, cluster_spec, zk_path, args):
    device = torch.device('cpu')
    model = LeNet()

    dataset = FEMNISTDataset(args.dataset_dir, download=args.dataset_download,
                             only_digits=True, transform=transforms.ToTensor())
    partitioner = FEMNISTDatasetPartitioner(
        dataset, len(cluster_spec), args.seed, args.max_num_users)
    partition = partitioner.get(rank-1)
    data_loader = torch.utils.data.DataLoader(
        partition, batch_size=args.batch_size, shuffle=True)

    worker = Worker(model, device, rank, cluster_spec, zk_path, args.zk_hosts)
    worker.init()
    worker.run(args.epochs, args.local_epochs, args.lr,
               data_loader, args.log_every_n_steps)
    worker.terminate()


DEFAULT_ARGS = {
    'epochs': 10,
    'local_epochs': 10,
    'batch_size': 32,
    'lr': 0.001,
    'log_every_n_steps': 10,
    'seed': 1234,
    'zk_hosts': '127.0.0.1:2181',
    'port': 12345,
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
        '--log_every_n_steps', type=int,
        default=DEFAULT_ARGS['log_every_n_steps'],
        help='log every n steps (default={})'.format(
            DEFAULT_ARGS['log_every_n_steps']))
    parser.add_argument(
        '--seed', type=int, default=DEFAULT_ARGS['seed'],
        help='random seed (default={})'.format(DEFAULT_ARGS['seed']))
    parser.add_argument(
        '--zk_hosts', default=DEFAULT_ARGS['zk_hosts'],
        help='Zookeeper hosts for leader election (default={})'.format(
            DEFAULT_ARGS['zk_hosts']))
    parser.add_argument(
        '--name',
        help='federated training session name to distinguish. '
             'uuid will be generated to use for this If not provided.')
    parser.add_argument(
        '--port', type=int, default=DEFAULT_ARGS['port'],
        help='base port number (default={})'.format(DEFAULT_ARGS['port']))
    parser.add_argument(
        '--max_num_users', type=int,
        help='max number of users that each worker can hold')

    args = parser.parse_args()

    cluster_spec = ['localhost:%d' % (args.port + i)
                    for i in range(args.num_workers)]

    session_name = args.name or uuid.uuid4().hex
    zk_path = '/election_' + session_name

    processes = []
    for rank in range(args.num_workers):
        p = mp.Process(target=run_worker,
                       args=(rank, cluster_spec, zk_path, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    ZkElection(0, path=zk_path).delete_path(recursive=True)


if __name__ == "__main__":
    main()
