from __future__ import absolute_import, division, print_function

import argparse
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import uuid

from datasets.femnist import FEMNISTDataset, FEMNISTDatasetPartitioner
from leader_based.worker import Worker
from leader_based.zk_election import ZkElection
from nets.lenet import LeNet
from utils.train import TrainArguments
import utils.flags as flags


def run_worker(rank, cluster_spec, zk_path, args):
    dataset = FEMNISTDataset(args.dataset_dir, download=args.dataset_download,
                             only_digits=True, transform=transforms.ToTensor())
    partitioner = FEMNISTDatasetPartitioner(
        dataset, len(cluster_spec), args.seed, args.max_num_users)
    partition = partitioner.get(rank)
    data_loader = torch.utils.data.DataLoader(
        partition, batch_size=args.batch_size, shuffle=True)

    validation = None
    if args.validation_period:
        test_dataset = FEMNISTDataset(args.dataset_dir, train=False,
                                      only_digits=True, transform=transforms.ToTensor())
        test_partitioner = FEMNISTDatasetPartitioner(
            test_dataset, len(cluster_spec), args.seed, args.max_num_users)
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

    device = torch.device('cpu')
    model = LeNet()
    train_args = TrainArguments(
        data_loader=data_loader,
        device=device,
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=args.lr),
        loss_fn=F.nll_loss,
        log_every_n_steps=args.log_every_n_steps,
    )

    worker = Worker(model, device, rank, cluster_spec,
                    zk_path, args.zk_hosts, admm_kwargs)
    worker.init()
    worker.run(args.epochs, args.local_epochs, train_args, validation)
    worker.terminate()


DEFAULT_ARGS = {
    'zk_hosts': '127.0.0.1:2181',
    'port': 12345,
}


def main():
    parser = argparse.ArgumentParser()
    flags.add_base_flags(parser)
    flags.add_admm_flags(parser)
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

    args = parser.parse_args()

    torch.manual_seed(args.seed)

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
