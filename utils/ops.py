from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F


def weighted_sum(tensors, weights):
    with torch.no_grad():
        weighted_tensors = torch.stack([
            weight * tensor
            for weight, tensor in zip(weights, tensors)])
        return torch.sum(weighted_tensors, dim=0)


def aggregate_state_dicts_by_names(state_dicts):
    retval = {}
    for state_dict in state_dicts:
        for name, params in state_dict.items():
            if name in retval:
                retval[name].append(params)
            else:
                retval[name] = [params]
    return retval


def calculate_mse(state_dict, other_dict):
    with torch.no_grad():
        flattened_params = torch.cat([p.flatten()
                                      for p in state_dict.values()])
        flattened_others = torch.cat([p.flatten()
                                      for p in other_dict.values()])
        return F.mse_loss(flattened_params, flattened_others)


def calculate_distance(params, others):
    with torch.no_grad():
        distance = 0.0
        num_elems = 0
        for param, other in zip(params, others):
            distance += torch.norm(param - other) ** 2
            num_elems += param.numel()
        return torch.sqrt(distance) / num_elems
