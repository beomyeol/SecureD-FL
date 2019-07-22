from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

import datasets.vehicle


class LinearSVM(nn.Module):

    def __init__(self):
        super(LinearSVM, self).__init__()
        self.linear = nn.Linear(100, 1)


    def forward(self, x):
        return self.linear(x)


def hinge_loss_fn(input, target):
    loss = 1.0 - torch.mul(input, target)
    loss[loss < 0] = 0
    return torch.mean(loss)

loss_fn = hinge_loss_fn
dataset = datasets.vehicle

def test_fn(output, target):
    correct = (torch.sign(output) == target).sum().item()
    total = target.size(0)
    return correct, total