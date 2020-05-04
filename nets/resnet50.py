from __future__ import absolute_import, division, print_function

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

import datasets.cifar10


class ResNet50(resnet.ResNet):

    def __init__(self):
        super(ResNet50, self).__init__(
            resnet.Bottleneck, [3, 4, 6, 3], num_classes=10,
            norm_layer=functools.partial(
                nn.BatchNorm2d, track_running_stats=False))

    def forward(self, x):
        x = super(ResNet50, self).forward(x)
        return F.log_softmax(x, dim=1)


loss_fn = F.nll_loss
dataset = datasets.cifar10


def test_fn(output, target):
    _, pred = torch.max(output.data, dim=1)
    return (pred == target).sum().item(), target.size(0)
