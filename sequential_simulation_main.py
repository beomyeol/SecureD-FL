from __future__ import absolute_import, division, print_function

import argparse
import copy
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import datasets.femnist as femnist
from datasets.partition import DatasetPartitioner
from nets.net_factory import create_net
from sequential.worker import Worker, aggregate_models
from utils.train import TrainArguments
from utils.test import TestArguments
import utils.flags as flags
import utils.logger as logger


_LOGGER = logger.get_logger(__file__)


DEFAULT_ARGS = {
    'gpu_id': 0,
    'model': 'lenet'
}


def run_clustering_based_aggreation(workers, num_clusters):
    from sequential.worker import run_clustering
    kmeans = run_clustering(workers, num_clusters)

    _LOGGER.info('clustering labels: %s', kmeans.labels_)

    worker_clusters = [[] for _ in range(num_clusters)]
    for worker, label in zip(workers, kmeans.labels_):
        worker_clusters[label].append(worker)

    # TODO: reconstruct parameters from kmeans.cluster_centers_
    for i, workers in enumerate(worker_clusters):
        # TODO: support weighted aggreagtion
        _LOGGER.info('cluster_id: %d, #workers: %d', i, len(workers))
        aggregated_state_dict = aggregate_models(workers)
        for worker in workers:
            worker.model.load_state_dict(aggregated_state_dict)


def run_simulation(workers, args):
    weights = None
    if args.adjust_local_epochs or args.weighted_avg:
        num_batches = []
        for worker in workers:
            num_batches.append(
                len(worker.train_args.data_loader.dataset))

        if args.adjust_local_epochs:
            lcm = np.lcm.reduce(num_batches)
            ratios = lcm / num_batches
            ratios *= args.local_epochs * len(workers) / np.sum(ratios)
            local_epochs_list = [int(round(local_epochs))
                                 for local_epochs in ratios]
            _LOGGER.info('local epochs: %s', str(local_epochs_list))

            for worker, local_epochs in zip(workers, local_epochs_list):
                if local_epochs == 0:
                    local_epochs = 1
                worker.local_epochs = local_epochs

        if args.weighted_avg:
            weights = np.array(num_batches) / np.sum(num_batches)
            _LOGGER.info('weights: %s', str(weights))

    for epoch in range(args.epochs):
        log_prefix = 'epoch: [{}/{}]'.format(epoch, args.epochs)
        elapsed_times = []
        for worker in workers:
            new_log_prefix = '{}, rank: {}'.format(log_prefix, worker.rank)
            t = time.time()
            worker.train(new_log_prefix)
            elapsed_time = time.time() - t
            _LOGGER.info('%s, comp_time: %f', new_log_prefix, elapsed_time)
            elapsed_times.append(elapsed_time)

        _LOGGER.info(log_prefix + ', elapsed_time: %f', max(elapsed_times))

        if not args.wo_sync:
            if args.num_clusters:
                run_clustering_based_aggreation(workers, args.num_clusters)
            else:
                aggregated_state_dict = aggregate_models(workers, weights)
                for worker in workers:
                    worker.model.load_state_dict(aggregated_state_dict)

        for worker in workers:
            new_log_prefix = '{}, rank: {}'.format(log_prefix, worker.rank)
            if worker.test_args and epoch % worker.test_args.period == 0:
                worker.test(new_log_prefix)


def main():
    parser = argparse.ArgumentParser()
    flags.add_base_flags(parser)
    flags.add_admm_flags(parser)
    parser.add_argument(
        '--adjust_local_epochs', action='store_true',
        help='adjust local epochs depending on # mini-batches')
    parser.add_argument(
        '--weighted_avg', action='store_true',
        help='enable the weighted avg based on # mini-batches')
    parser.add_argument(
        '--gpu_id', type=int, default=DEFAULT_ARGS['gpu_id'],
        help='gpu id to use (default={})'.format(DEFAULT_ARGS['gpu_id']))
    parser.add_argument(
        '--model', default=DEFAULT_ARGS['model'],
        help='name of ML model to train (default={})'.format(
            DEFAULT_ARGS['model']))
    parser.add_argument(
        '--wo_sync', action='store_true',
        help='run test without aggregation')
    parser.add_argument(
        '--num_clusters', type=int,
        help='use kmeans clustering among worker models for the given #clusters'
             ' and average models within each cluster')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    world_size = args.num_workers

    net_args = create_net(args.model, batch_size=args.batch_size)
    dataset = net_args.dataset
    partition_kwargs = net_args.partition_kwargs
    model = net_args.model
    train_fn = net_args.train_fn

    if torch.cuda.is_available():
        device = torch.device('cuda:%d' % args.gpu_id)
    else:
        device = torch.device('cpu')

    workers = []
    for rank in range(world_size):
        partition = dataset.get_partition(rank=rank,
                                          world_size=world_size,
                                          ratios=args.split_ratios,
                                          **partition_kwargs,
                                          **vars(args))
        _LOGGER.info('rank: %d, #clients: %d', rank, len(partition.client_ids))
        data_loader = torch.utils.data.DataLoader(
            partition, batch_size=args.batch_size, shuffle=True)

        new_model = copy.deepcopy(model).to(device)
        train_args = TrainArguments(
            data_loader=data_loader,
            device=device,
            model=new_model,
            optimizer=optim.Adam(new_model.parameters(), lr=args.lr),
            loss_fn=F.nll_loss,
            log_every_n_steps=args.log_every_n_steps,
            train_fn=train_fn,
        )

        local_epochs = args.local_epochs

        test_args = None
        if args.validation_period:
            test_partition = dataset.get_partition(rank=rank,
                                                   world_size=args.num_workers,
                                                   ratios=args.split_ratios,
                                                   train=False,
                                                   **partition_kwargs,
                                                   **vars(args))
            assert partition.client_ids == test_partition.client_ids
            test_data_loader = torch.utils.data.DataLoader(
                test_partition, batch_size=args.batch_size)
            test_args = TestArguments(
                data_loader=test_data_loader,
                model=new_model,
                device=device,
                period=args.validation_period)

        workers.append(Worker(rank, local_epochs, train_args, test_args))

    run_simulation(workers, args)


if __name__ == "__main__":
    main()
