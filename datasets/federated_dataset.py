from __future__ import absolute_import, division, print_function


class FederatedDataset(object):

    def client_ids(self):
        return NotImplementedError()

    def get_client_dataset(self, client_id):
        raise NotImplementedError()