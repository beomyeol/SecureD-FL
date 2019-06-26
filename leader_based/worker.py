from __future__ import absolute_import, division, print_function

import time
import torch.optim as optim
import torch.nn.functional as F

from network.network_manager import NetworkManager
from leader_based.zk_election import ZkElection
from leader_based.role import Leader, Follower, ADMMLeader, ADMMFollower
from utils import logger
from utils.train import train_single_epoch
from utils.test import test_model

_LOGGER = logger.get_logger(__file__)


class Worker(object):

    def __init__(self, model, device, rank, cluster_spec, zk_path, zk_hosts, admm_kwargs=None):
        self.model = model
        self.device = device
        self.rank = rank
        self.cluster_spec = cluster_spec
        self.zk_path = zk_path
        self.zk_hosts = zk_hosts
        self.admm_kwargs = admm_kwargs

        self.election = None
        self.role = None

    def init(self):
        network_mgr = NetworkManager(self.rank, self.cluster_spec)
        network_mgr.start_server()

        self.election = ZkElection(
            self.rank, path=self.zk_path, hosts=self.zk_hosts)
        is_leader = self.election.run()

        # wait until all workers are online
        while len(self.election.get_online_workers()) < len(self.cluster_spec):
            time.sleep(0.1)

        _LOGGER.info('rank=%d, is_leader=%s', self.rank, str(is_leader))

        if self.admm_kwargs:
            if is_leader:
                role = ADMMLeader(self.rank, network_mgr, **self.admm_kwargs)
            else:
                role = ADMMFollower(
                    self.rank, self.election.get_leader_rank(), network_mgr, **self.admm_kwargs)
        else:
            if is_leader:
                role = Leader(self.rank, network_mgr)
            else:
                role = Follower(
                    self.rank, self.election.get_leader_rank(), network_mgr)

        self.role = role

    def run(self, epochs, local_epochs, lr, data_loader, log_every_n_steps, validation):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = F.nll_loss

        validation_period = None
        validation_loader = None
        if validation:
            validation_period, validation_loader = validation

        for epoch in range(epochs):
            log_prefix = '[worker] rank: {}, epoch: [{}/{}]'.format(
                self.rank, epoch, epochs)
            self.role.begin(self.model)
            for local_epoch in range(local_epochs):
                new_log_prefix = '{}, local_epoch: [{}/{}]'.format(
                    log_prefix, local_epoch, local_epochs)
                train_single_epoch(
                    data_loader, self.model, optimizer, loss_fn,
                    log_every_n_steps, self.device, new_log_prefix)
            self.role.end(self.model)

            if validation_period and epoch % validation_period == 0:
                test_model(validation_loader, self.model, self.device, log_prefix)

        if isinstance(self.role, ADMMLeader):
            _LOGGER.info('Avg ADMM iteration: %s', self.role.total_iter/epochs)

    def terminate(self):
        self.role.terminate()
        self.election.terminate()
