from __future__ import absolute_import, division, print_function

import time

from network.network_manager import NetworkManager
from leader_based.zk_election import ZkElection
from leader_based.role import Leader, Follower, ADMMLeader, ADMMFollower
from utils import logger
from utils.test import test_model

_LOGGER = logger.get_logger(__file__)


class Worker(object):

    def __init__(self, rank, cluster_spec, zk_path, zk_hosts, op, admm_kwargs=None):
        self.rank = rank
        self.cluster_spec = cluster_spec
        self.zk_path = zk_path
        self.zk_hosts = zk_hosts
        self.op = op
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
                role = Leader(self.rank, network_mgr, self.op)
            else:
                role = Follower(
                    self.rank, self.election.get_leader_rank(), network_mgr)

        self.role = role

    def run(self, epochs, local_epochs, train_args, test_args):
        for epoch in range(epochs):
            log_prefix = '[worker] rank: {}, epoch: [{}/{}]'.format(
                self.rank, epoch, epochs)
            self.role.begin(train_args.model)
            for local_epoch in range(local_epochs):
                new_log_prefix = '{}, local_epoch: [{}/{}]'.format(
                    log_prefix, local_epoch, local_epochs)
                train_args.train_fn(train_args, new_log_prefix)
            self.role.end(train_args.model)

            if test_args and epoch % test_args.period == 0:
                # synchronization with the expectation that role.begin() would do
                # TODO: avoid dupplicate execution of role.begin()
                self.role.begin(train_args.model)
                test_model(test_args, log_prefix)

        if isinstance(self.role, ADMMLeader):
            _LOGGER.info('Avg ADMM iteration: %s', self.role.total_iter/epochs)

    def terminate(self):
        self.role.terminate()
        self.election.terminate()
