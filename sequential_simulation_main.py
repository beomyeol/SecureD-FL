from __future__ import absolute_import, division, print_function

import argparse
import copy
import time
import torch
import torch.optim as optim
import numpy as np
import os
import math

from datasets.partition import DatasetPartitioner
from nets.net_factory import create_net
from sequential.worker import Worker, aggregate_models
from utils.train import TrainArguments
from utils.test import TestArguments
import utils.flags as flags
import utils.logger as logger


_LOGGER = logger.get_logger(__file__)


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


def is_perfect_square(n):
    sqrt = math.sqrt(n)
    return sqrt == int(sqrt)


def create_non_overlapping_groups(num_workers):
    if num_workers < 9 or not is_perfect_square(num_workers):
        raise ValueError('Invalid # workers: %d' % num_workers)

    num_groups = num_elems = int(math.sqrt(num_workers))
    groups1 = [list(range(i * num_elems, (i+1) * num_elems))
               for i in range(num_groups)]
    groups2 = [[] for _ in range(num_groups)]
    for i in range(num_workers):
        groups2[i % num_groups].append(i)
    return groups1, groups2


def run_aggregation(workers, weights, args):
    if args.num_clusters:
        run_clustering_based_aggreation(workers, args.num_clusters)
    else:
        admm_kwargs = None
        if args.use_admm:
            admm_kwargs = {
                'max_iter': args.admm_max_iter,
                'threshold': args.admm_threshold,
                'lr': args.admm_lr,
                'decay_period': args.admm_decay_period,
                'decay_rate': args.admm_decay_rate,
            }
            if args.secure_admm:
                admm_kwargs['groups_pair'] = create_non_overlapping_groups(
                    len(workers))
        aggregated_state_dict = aggregate_models(workers, weights, admm_kwargs)
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

    for epoch in range(1, args.epochs + 1):
        log_prefix = 'epoch: [{}/{}]'.format(epoch, args.epochs)
        elapsed_times = []

        if epoch > 1:
            run_aggregation(workers, weights, args)

        for worker in workers:
            new_log_prefix = '{}, rank: {}'.format(log_prefix, worker.rank)
            t = time.time()
            worker.train(new_log_prefix)
            elapsed_time = time.time() - t
            _LOGGER.info('%s, comp_time: %f, mean_loss: %f',
                         new_log_prefix,
                         elapsed_time,
                         np.mean(worker.losses))
            elapsed_times.append(elapsed_time)

        _LOGGER.info(log_prefix + ', elapsed_time: %f', max(elapsed_times))

        for worker in workers:
            new_log_prefix = '{}, rank: {}'.format(log_prefix, worker.rank)
            if worker.test_args and epoch % worker.test_args.period == 0:
                worker.test(new_log_prefix)

        if args.save_period and epoch % args.save_period == 0:
            save_dict = {worker.rank: worker.model.state_dict()
                         for worker in workers}
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, '%d.ckpt' % epoch)
            _LOGGER.info('saving the model states to %s...', save_path)
            torch.save(save_dict, save_path)

        if epoch == args.epochs and args.validation_period:
            # last epoch
            # run aggregation and measure test epoch
            run_aggregation(workers, weights, args)
            _LOGGER.info('test after aggregation')
            for worker in workers:
                new_log_prefix = '{}, rank: {}'.format(log_prefix, worker.rank)
                if worker.test_args and epoch % worker.test_args.period == 0:
                    worker.test(new_log_prefix)


def check_args_validity(args):
    flags.check_admm_args(args)
    if args.save_period:
        assert args.save_period > 0
        assert args.save_dir
    if args.secure_admm:
        args.num_workers >= 9, '#workers should be >= 9'
        is_perfect_square(args.num_workers), \
            '#workers should be squares for secure admm'


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
        '--gpu_id', type=int, help='gpu id to use')
    parser.add_argument(
        '--model', required=True, help='name of ML model to train')
    parser.add_argument(
        '--num_clusters', type=int,
        help='use kmeans clustering among worker models for the given #clusters'
             ' and average models within each cluster')
    parser.add_argument(
        '--save_dir',
        help='save model states to the given path')
    parser.add_argument(
        '--save_period', type=int,
        help='save model states in every given epoch')
    parser.add_argument(
        '--secure_admm', action='store_true', help='use secure admm')

    args = parser.parse_args()
    check_args_validity(args)

    _LOGGER.info('Seed: %d', args.seed)

    torch.manual_seed(args.seed)

    if args.gpu_id is not None:
        device = torch.device('cuda:%d' % args.gpu_id)
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(args.seed)
        _LOGGER.info('Using cuda id=%d', torch.cuda.current_device())
    else:
        device = torch.device('cpu')

    net_args = create_net(args.model, batch_size=args.batch_size)
    load_dataset_fn = net_args.load_dataset_fn
    model = net_args.model
    train_fn = net_args.train_fn
    test_fn = net_args.test_fn
    loss_fn = net_args.loss_fn

    dataset = load_dataset_fn(train=True, **vars(args))
    _LOGGER.info('#clients in the dataset: %d', len(dataset.client_ids()))

    if args.num_workers == -1:
        world_size = len(dataset.client_ids())
    else:
        world_size = args.num_workers

    _LOGGER.info('world_size: %d', world_size)
    partitioner = DatasetPartitioner(dataset, world_size, args.split_ratios,
                                     args.seed, args.max_num_users_per_worker)

    if args.validation_period:
        test_dataset = load_dataset_fn(train=False, **vars(args))
        test_partitioner = DatasetPartitioner(
            test_dataset, world_size, args.split_ratios, args.seed,
            args.max_num_users_per_worker)

    workers = []
    for rank in range(world_size):
        partition = partitioner.get(rank)
        _LOGGER.info('rank: %d, #clients: %d', rank, len(partition.client_ids))
        data_loader = torch.utils.data.DataLoader(
            partition, batch_size=args.batch_size, shuffle=True)

        new_model = copy.deepcopy(model).to(device)
        train_args = TrainArguments(
            data_loader=data_loader,
            device=device,
            model=new_model,
            optimizer=optim.Adam(new_model.parameters(), lr=args.lr),
            loss_fn=loss_fn,
            log_every_n_steps=args.log_every_n_steps,
            train_fn=train_fn,
        )

        local_epochs = args.local_epochs

        test_args = None
        if args.validation_period:
            test_dataset = load_dataset_fn(**vars(args))
            test_partition = test_partitioner.get(rank)
            assert partition.client_ids == test_partition.client_ids
            test_data_loader = torch.utils.data.DataLoader(
                test_partition, batch_size=args.batch_size)
            test_args = TestArguments(
                data_loader=test_data_loader,
                model=new_model,
                device=device,
                period=args.validation_period,
                test_fn=test_fn)

        workers.append(Worker(rank, local_epochs, train_args, test_args))

    run_simulation(workers, args)


if __name__ == "__main__":
    main()
