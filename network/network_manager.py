from __future__ import absolute_import, division, print_function

import socket
import struct
import time
import threading
import io
import queue
from socketserver import ThreadingTCPServer, BaseRequestHandler

from utils import logger

_LOGGER = logger.get_logger(__file__)
Empty = queue.Empty


class Handler(BaseRequestHandler):

    def __init__(self, queue, *args, **kwargs):
        self.queue = queue
        super(Handler, self).__init__(*args, **kwargs)

    def handle(self):
        with self.request.makefile(mode='rb') as f:
            while True:
                length_bytes = f.read(4)
                if len(length_bytes) < 4:
                    break
                bytes_to_read = struct.unpack('>I', length_bytes)[0]
                buffer = io.BytesIO()

                while bytes_to_read > 0:
                    msg_bytes = f.read(bytes_to_read)
                    if len(msg_bytes):
                        buffer.write(msg_bytes)
                        bytes_to_read -= len(msg_bytes)
                    else:
                        break

                if bytes_to_read == 0:
                    self.queue.put(buffer.getvalue())
                else:
                    break


class NetworkManager(object):

    def __init__(self, rank, cluster_spec):
        self.rank = rank
        self.cluster_spec = cluster_spec

        self.sockets = [None for _ in range(len(self.cluster_spec))]
        self.msg_queue = queue.LifoQueue()

        def handler_factory(*args, **kwargs):
            return Handler(self.msg_queue, *args, **kwargs)

        port = int(cluster_spec[rank].split(':')[1])
        self.server = ThreadingTCPServer(
            ('0.0.0.0', port), handler_factory, bind_and_activate=False)

    def start_server(self):
        self.server.socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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

    def recv(self, block=True):
        return self.msg_queue.get(block=block)

    def terminate(self):
        for socket in self.sockets:
            if socket:
                socket.close()
        self.stop_server()
