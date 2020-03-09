from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

import datasets.cifar10


class ResNet18(resnet.ResNet):

    def __init__(self):
        super(ResNet18, self).__init__(
            resnet.BasicBlock, [2, 2, 2, 2], num_classes=10)

    def forward(self, x):
        x = super(ResNet18, self).forward(x)
        return F.log_softmax(x, dim=1)


loss_fn = F.nll_loss
dataset = datasets.cifar10


def test_fn(output, target):
    _, pred = torch.max(output.data, dim=1)
    return (pred == target).sum().item(), target.size(0)
