from __future__ import absolute_import, division, print_function


class MockModel(object):

    def __init__(self, state_dict, device):
        self._state_dict = {name: parameter.to(device)
                            for name, parameter in state_dict.items()}

    def parameters(self):
        for parameter in self._state_dict.values():
            yield parameter

    def named_parameters(self):
        for name, parameter in self._state_dict.items():
            yield name, parameter

    def state_dict(self):
        return self._state_dict
