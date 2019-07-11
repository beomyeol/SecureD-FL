from __future__ import absolute_import, division, print_function

import argparse
import copy
import time
import torch
import torch.optim as optim
import torch.nn.functional as F

import datasets.femnist as femnist
from datasets.partition import DatasetPartitioner
from nets.lenet import LeNet
from utils.train import TrainArguments, train_model
from utils.test import TestArguments, test_model
import utils.flags as flags
import utils.logger as logger


_LOGGER = logger.get_logger(__file__)


class Worker(object):

    def __init__(self, rank, local_epochs, train_args, test_args):
        self.rank = rank
        self.local_epochs = local_epochs
        self.train_args = train_args
        self.test_args = test_args

    def train(self, log_prefix):
        for local_epoch in range(self.local_epochs):
            new_log_prefix = '{}, local_epoch: [{}/{}]'.format(
                log_prefix, local_epoch, self.local_epochs)
            self.train_args.train_fn(
                self.train_args, log_prefix=new_log_prefix)

    def test(self, log_prefix):
        test_model(self.test_args, log_prefix)

    @property
    def model(self):
        return self.train_args.model


def aggregate_models(workers, weights):
    with torch.no_grad():
        aggregated_state_dict = {}
        for worker, weight in zip(workers, weights):
            for name, parameter in worker.model.named_parameters():
                tensor = weight * parameter.data
                if name in aggregated_state_dict:
                    aggregated_state_dict[name] += tensor
                else:
                    aggregated_state_dict[name] = tensor

    return aggregated_state_dict


def main():
    parser = argparse.ArgumentParser()
    flags.add_base_flags(parser)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    world_size = args.num_workers

    dataset = femnist
    partition_kwargs = {}
    model = LeNet()
    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    train_fn = train_model

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

    weights = [1/world_size] * world_size

    for epoch in range(args.epochs):
        log_prefix = 'epoch: [{}/{}]'.format(epoch, args.epochs)
        for worker in workers:
            new_log_prefix = '{}, rank: {}'.format(log_prefix, worker.rank)
            t = time.time()
            worker.train(new_log_prefix)
            _LOGGER.info('%s, comp_time: %f', new_log_prefix, time.time() - t)

            if worker.test_args and epoch % worker.test_args.period == 0:
                worker.test(new_log_prefix)

        aggregated_state_dict = aggregate_models(workers, weights)

        for worker in workers:
            worker.model.load_state_dict(aggregated_state_dict)


if __name__ == "__main__":
    main()
