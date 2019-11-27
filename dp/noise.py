from __future__ import absolute_import, division, print_function

import torch
import math


class Gaussian(object):

    def __init__(self, eps, delta, sensitivity):
        self.eps = eps
        self.delta = delta
        self.sensitivity = sensitivity

        self.stdev = sensitivity / eps * math.sqrt(2 * math.log(1.25 / delta))

        self.normal_dist = torch.distributions.normal.Normal(
            loc=0, scale=self.stdev)

    def sample(self, shape=torch.Size([])):
        return self.normal_dist.sample(shape)


class AddNoise(object):

    def __init__(self, eps, delta, sensitivity):
        self.noise_gen = Gaussian(eps, delta, sensitivity)

    def __call__(self, tensor):
        return torch.add(tensor, self.noise_gen.sample(tensor.shape))
