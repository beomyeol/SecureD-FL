from __future__ import absolute_import, division, print_function

import socket
import struct
import threading
import queue
from socketserver import TCPServer, BaseRequestHandler

from utils import logger

_LOGGER = logger.get_logger(__file__)


class Handler(BaseRequestHandler):

    def __init__(self, queue, *args, **kwargs):
        self.queue = queue
        super(Handler, self).__init__(*args, **kwargs)

    def handle(self):
        length = struct.unpack('>I', self.request.recv(4))[0]
        msg_bytes = self.request.recv(length)
        self.queue.put(msg_bytes)


class NetworkManager(object):

    def __init__(self, rank, cluster_spec):
        self.rank = rank
        self.cluster_spec = cluster_spec

        self.sockets = [None for _ in range(len(self.cluster_spec))]
        self.msg_queue = queue.LifoQueue()

        def handler_factory(*args, **kwargs):
            return Handler(self.msg_queue, *args, **kwargs)

        port = int(cluster_spec[rank].split(':')[1])
        self.server = TCPServer(
            ('0.0.0.0', port), handler_factory, bind_and_activate=False)

    def start_server(self):
        self.server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.server_bind()
        self.server.server_activate()

        server_thread = threading.Thread(target=self.server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

    def stop_server(self):
        self.server.shutdown()

    def _connect(self, dst):
        if self.sockets[dst]:
            raise RuntimeError('socket to %d is already established' % dst)
        self.sockets[dst] = socket.create_connection(
            self.cluster_spec[dst].split(':'))

    def send(self, dst, data):
        if dst == self.rank:
            raise ValueError('Sending data to itself')

        if not self.sockets[dst]:
            self._connect(dst)

        data = struct.pack('>I', len(data)) + data
        self.sockets[dst].sendall(data)

    def broadcast(self, data):
        for i in range(len(self.cluster_spec)):
            if i != self.rank:
                self.send(i, data)

    def recv(self):
        return self.msg_queue.get()
