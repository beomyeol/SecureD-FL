from __future__ import absolute_import, division, print_function

import argparse
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
import uuid

import datasets.femnist as femnist
from leader_based.worker import Worker
from leader_based.role import Op
from leader_based.zk_election import ZkElection
from nets.lenet import LeNet
from utils.train import TrainArguments, train_model
from utils.test import TestArguments
import utils.flags as flags


def run_worker(rank, cluster_spec, zk_path, args):
    device = torch.device('cpu')
    model = LeNet()

    partition = femnist.get_partition(rank=rank,
                                      world_size=len(cluster_spec),
                                      **vars(args))
    data_loader = torch.utils.data.DataLoader(
        partition, batch_size=args.batch_size, shuffle=True)

    test_args = None
    if args.validation_period:
        test_partition = femnist.get_partition(rank=rank,
                                               world_size=len(cluster_spec),
                                               train=False,
                                               **vars(args))
        assert partition.client_ids == test_partition.client_ids
        test_data_loader = torch.utils.data.DataLoader(
            test_partition, batch_size=args.batch_size)
        test_args = TestArguments(
            data_loader=test_data_loader,
            model=model,
            device=device,
            period=args.validation_period)

    admm_kwargs = None
    if args.use_admm:
        admm_kwargs = {
            'max_iter': args.admm_max_iter,
            'threshold': args.admm_threshold,
            'lr': args.admm_lr
        }

    train_args = TrainArguments(
        data_loader=data_loader,
        device=device,
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=args.lr),
        loss_fn=F.nll_loss,
        log_every_n_steps=args.log_every_n_steps,
        train_fn=train_model,
    )

    worker = Worker(rank, cluster_spec, zk_path, args.zk_hosts,
                    args.op, admm_kwargs)
    worker.init()
    worker.run(args.epochs, args.local_epochs, train_args, test_args)
    worker.terminate()


DEFAULT_ARGS = {
    'zk_hosts': '127.0.0.1:2181',
    'port': 12345,
    'op': Op.MEAN,
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
    parser.add_argument(
        '--op', type=Op, choices=list(Op), default=DEFAULT_ARGS['op'],
        help='aggregation op at the leader (default={})'.format(DEFAULT_ARGS['op']))

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
