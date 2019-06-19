from __future__ import absolute_import, division, print_function

from kazoo.client import KazooClient


class ZkElection(object):
    # Should implement watcher

    _DELIM = '___'

    def __init__(self, rank, path='/election', hosts='127.0.0.1:2181',
                 timeout=2):
        self.rank = rank
        self.path = path
        self.create_path = self.path + '/' + str(rank) + self._DELIM
        self.zk = KazooClient(hosts=hosts, timeout=timeout)
        self.node = None

        self._ensured_path = False
        self._is_executed = False

        self.zk.start()

    def _get_rank(self, name):
        return int(name[:name.find(self._DELIM)])

    def _ensure_path(self):
        self.zk.ensure_path(self.path)
        self._ensured_path = True

    def _get_sorted_children(self):
        children = self.zk.get_children(self.path)

        def _seq(name):
            return name[name.find(self._DELIM) + len(self._DELIM):]

        children.sort(key=_seq)
        return children

    def get_online_workers(self):
        if not self._ensured_path:
            self._ensure_path()

        children = self._get_sorted_children()
        # strip sequence numbers
        ranks = [self._get_rank(child) for child in children]
        return ranks

    def run(self):
        if not self._ensured_path:
            self._ensure_path()

        if not self._is_executed:
            node = self.zk.create(
                self.create_path, ephemeral=True, sequence=True)
            self.node = node[len(self.path) + 1:]
            self._is_executed = True

        return self.is_leader()

    def _get_leader_node(self):
        if not self._is_executed:
            raise RuntimeError('Election is not executed.')

        return self._get_sorted_children()[0]

    def get_leader_rank(self):
        return self._get_rank(self._get_leader_node())

    def is_leader(self):
        return self.node == self._get_leader_node()

    def terminate(self):
        self.zk.stop()

    def delete_path(self, recursive=False):
        self.zk.delete(self.path, recursive=recursive)
