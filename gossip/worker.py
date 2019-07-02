from __future__ import absolute_import, division, print_function

import torch
import random
import io

from network.network_manager import NetworkManager, Empty
from utils import logger
from utils.train import train_single_epoch
from utils.test import test_model

_LOGGER = logger.get_logger(__file__, level=logger.INFO)


class Message(object):

    def __init__(self, src_rank, epoch, model_dict):
        self.src_rank = src_rank
        self.epoch = epoch
        self.model_dict = model_dict


class Worker(object):

    def __init__(self, rank, cluster_spec, num_gossips, seed=None, admm_kwargs=None):
        self.rank = rank
        self.cluster_spec = cluster_spec
        self.num_gossips = num_gossips
        self.rng = random.Random()
        self.rng.seed(seed)
        self.seed = seed
        self.admm_kwargs = admm_kwargs

        self.candidate_ranks = list(range(len(cluster_spec)))
        del self.candidate_ranks[self.rank]

        self.network_mgr = NetworkManager(self.rank, self.cluster_spec)
        self.network_mgr.start_server()

    def run(self, epochs, local_epochs, train_args, validation=(None, None)):
        validation_period, validation_loader = validation
        # CAVEATS: assume that model parameters of all workers are the same at the beginning.
        # TODO: is this assumption necessary?

        for epoch in range(epochs):
            log_prefix = '[worker] rank: {}, epoch: [{}/{}]'.format(
                self.rank, epoch, epochs)
            for local_epoch in range(local_epochs):
                new_log_prefix = '{}, local_epoch: [{}/{}]'.format(
                    log_prefix, local_epoch, local_epochs)
                train_single_epoch(train_args, new_log_prefix)

            self.run_gossip(epoch, train_args.model)

            if validation_period and epoch % validation_period == 0:
                test_model(validation_loader, train_args.model,
                           train_args.device, log_prefix)

    def terminate(self):
        self.network_mgr.terminate()

    def run_gossip(self, epoch, model):
        # send gossip
        target_ranks = self.rng.sample(self.candidate_ranks, self.num_gossips)

        model_dict = model.state_dict()
        buffer = io.BytesIO()
        msg = Message(self.rank, epoch, model_dict)
        torch.save(msg, buffer)
        msg_bytes = buffer.getvalue()

        for target_rank in target_ranks:
            try:
                self.network_mgr.send(target_rank, msg_bytes)
            except Exception as e:
                _LOGGER.warn('failed to send to %d: %s', target_rank, e)

        # process received gossips
        received_model_dicts = []
        try:
            while True:
                msg_bytes = self.network_mgr.recv(block=False)
                msg = torch.load(io.BytesIO(msg_bytes))
                _LOGGER.debug('received model. rank=%d, epoch=%d',
                              msg.src_rank, msg.epoch)
        except Empty:
            pass

        for received_model_dict in received_model_dicts:
            for name in model_dict.keys():
                model_dict[name] += received_model_dict[name]

        for parameter in model_dict.values():
            parameter.data /= len(received_model_dicts) + 1

        # update the model with the aggregated one
        model.load_state_dict(model_dict)
