from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import datasets.cifar10


class CifarNet(nn.Module):

    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.lrn = nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1.0)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        x = self.lrn(self.pool(F.relu(self.conv1(x))))
        x = self.pool(self.lrn(F.relu(self.conv2(x))))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


loss_fn = F.nll_loss
dataset = datasets.cifar10


def test_fn(output, target):
    _, pred = torch.max(output.data, dim=1)
    return (pred == target).sum().item(), target.size(0)
