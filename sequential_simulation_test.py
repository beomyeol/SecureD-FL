"""Sequential Simulation Tests."""
# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name
from __future__ import absolute_import, division, print_function

import unittest
from unittest.mock import Mock

import numpy.testing as npt
import torch.nn as nn

import utils.ops as ops
from sequential.worker import *
from sequential_simulation_main import create_non_overlapping_groups


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.fc = nn.Linear(2, 2)


def create_test_model(weights, bias):
    model = TestModel()
    state_dict = {
        'fc.weight': torch.tensor(weights),
        'fc.bias': torch.tensor(bias),
    }
    model.load_state_dict(state_dict)
    return model


def generate_test_models(num_models):
    models = []
    for _ in range(num_models):
        weight = torch.rand(2, 2).tolist()
        bias = torch.rand(2).tolist()
        models.append(create_test_model(weight, bias))
    return models


class TestAggregation(unittest.TestCase):

    def test_uniform_aggregation(self):
        weight1 = [[0.1, 0.2],
                   [0.3, 0.4]]
        weight2 = [[0.1, 0.2],
                   [0.3, 0.4]]
        bias1 = [0.1, 0.2]
        bias2 = [0.1, 0.2]
        expected_weight = [[0.1, 0.2],
                           [0.3, 0.4]]
        expected_bias = [0.1, 0.2]
        model1 = create_test_model(weight1, bias1)
        model2 = create_test_model(weight2, bias2)

        workers = [
            Mock(model=model1),
            Mock(model=model2),
        ]

        expected_params = [expected_weight, expected_bias]
        state_dict = aggregate_models(workers)

        for param, expected_param in zip(state_dict.values(), expected_params):
            npt.assert_almost_equal(param.tolist(), expected_param)

    def test_weighted_aggregation(self):
        weight1 = [[0.1, 0.2],
                   [0.3, 0.4]]
        weight2 = [[-0.1, -0.2],
                   [-0.3, -0.4]]
        bias1 = [0.1, 0.2]
        bias2 = [-0.1, -0.2]
        aggr_weights = [0.8, 0.2]
        expected_weight = [[0.06, 0.12],
                           [0.18, 0.24]]
        expected_bias = [0.06, 0.12]
        model1 = create_test_model(weight1, bias1)
        model2 = create_test_model(weight2, bias2)

        workers = [
            Mock(model=model1),
            Mock(model=model2),
        ]

        expected_params = [expected_weight, expected_bias]
        state_dict = aggregate_models(workers, weights=aggr_weights)

        for param, expected_param in zip(state_dict.values(), expected_params):
            npt.assert_almost_equal(param.tolist(), expected_param)

    def test_uniform_admm_aggregation(self):
        weight1 = [[0.1, 0.2],
                   [0.3, 0.4]]
        weight2 = [[0.1, 0.2],
                   [0.3, 0.4]]
        bias1 = [0.1, 0.2]
        bias2 = [0.1, 0.2]
        expected_weight = [[0.1, 0.2],
                           [0.3, 0.4]]
        expected_bias = [0.1, 0.2]
        admm_kwargs = {
            'max_iter': 10,
            'threshold': 5e-3,
            'lr': 0.01,
            'decay_period': 2,
            'decay_rate': 0.5,
        }

        device = torch.device('cpu')
        model1 = create_test_model(weight1, bias1)
        model2 = create_test_model(weight2, bias2)

        workers = [
            Mock(model=model1, device=device),
            Mock(model=model2, device=device),
        ]

        expected_params = [expected_weight, expected_bias]
        state_dict = aggregate_models(workers, admm_kwargs=admm_kwargs)

        for param, expected_param in zip(state_dict.values(), expected_params):
            npt.assert_almost_equal(param.tolist(), expected_param, decimal=3)

    def test_weighted_admm_aggregation(self):
        weight1 = [[0.1, 0.2],
                   [0.3, 0.4]]
        weight2 = [[-0.1, -0.2],
                   [-0.3, -0.4]]
        bias1 = [0.1, 0.2]
        bias2 = [-0.1, -0.2]
        aggr_weights = [0.8, 0.2]
        expected_weight = [[0.06, 0.12],
                           [0.18, 0.24]]
        expected_bias = [0.06, 0.12]
        admm_kwargs = {
            'max_iter': 10,
            'threshold': 5e-3,
            'lr': 0.01,
            'decay_period': 2,
            'decay_rate': 0.5,
        }

        device = torch.device('cpu')
        model1 = create_test_model(weight1, bias1)
        model2 = create_test_model(weight2, bias2)

        workers = [
            Mock(model=model1, device=device),
            Mock(model=model2, device=device),
        ]

        expected_params = [expected_weight, expected_bias]
        state_dict = aggregate_models(
            workers, weights=aggr_weights, admm_kwargs=admm_kwargs)

        for param, expected_param in zip(state_dict.values(), expected_params):
            npt.assert_almost_equal(param.tolist(), expected_param, decimal=3)


class TestClustering(unittest.TestCase):

    def test_clustering(self):
        weight1 = [[0.1, 0.2],
                   [0.3, 0.4]]
        weight2 = [[-0.1, -0.2],
                   [-0.3, -0.4]]
        weight3 = [[0.2, 0.3],
                   [0.4, 0.5]]
        bias1 = [0.1, 0.2]
        bias2 = [-0.1, -0.2]
        bias3 = [0.2, 0.3]

        model1 = create_test_model(weight1, bias1)
        model2 = create_test_model(weight2, bias2)
        model3 = create_test_model(weight3, bias3)

        workers = [
            Mock(model=model1),
            Mock(model=model2),
            Mock(model=model3),
        ]

        kmeans = run_clustering(workers, 2)
        labels = kmeans.labels_

        self.assertEqual(labels[0], labels[2])
        self.assertNotEqual(labels[0], labels[1])


class TestFedAvg(unittest.TestCase):

    def test_fedavg(self):
        num_models = 10
        models = generate_test_models(num_models)

        result_state_dict = fedavg(models)

        with torch.no_grad():
            aggregated_state_dict = ops.aggregate_state_dicts_by_names(
                [model.state_dict() for model in models])
            expected_state_dict = {
                name: torch.mean(torch.stack(parameters), dim=0)
                for name, parameters
                in aggregated_state_dict.items()}

        for name in expected_state_dict:
            npt.assert_almost_equal(result_state_dict[name].tolist(),
                                    expected_state_dict[name].tolist())


class TestSecureADMM(unittest.TestCase):

    def test_non_overlapping_groups_with_less_num_workers(self):
        for i in range(8):
            self.assertRaises(ValueError, create_non_overlapping_groups, i)

    def test_non_overlapping_groups_with_non_perpect_square(self):
        for i in range(10, 16):
            self.assertRaises(ValueError, create_non_overlapping_groups, i)

    def test_non_overlapping_groups(self):
        for num_workers in [9, 16, 25]:
            groups1, groups2 = create_non_overlapping_groups(num_workers)

            def find_group(rank, groups):
                for group in groups:
                    if rank in group:
                        return group

            for i in range(num_workers):
                group1 = find_group(i, groups1)
                group2 = find_group(i, groups2)
                intersection = set(group1).intersection(set(group2))
                self.assertEqual(intersection, {i})

    def test_secure_uniform_aggregation(self):
        num_workers = 9
        weight_list = [[[0.1, 0.2], [0.3, 0.4]],
                       [[0.2, 0.4], [0.6, 0.8]],
                       [[0.3, 0.6], [0.9, 1.2]]]
        bias_list = [[0.1, 0.2], [0.2, 0.4], [0.3, 0.6]]
        expected_weight = [[0.2, 0.4], [0.6, 0.8]]
        expected_bias = [0.2, 0.4]
        admm_kwargs = {
            'max_iter': 10,
            'threshold': 1e-5,
            'lr': 0.01,
            'decay_period': 2,
            'decay_rate': 0.5,
            'groups': create_non_overlapping_groups(num_workers),
        }

        device = torch.device('cpu')
        workers = [
            Mock(model=create_test_model(weight_list[i % 3], bias_list[i % 3]),
                 device=device)
            for i in range(num_workers)]

        expected_params = [expected_weight, expected_bias]
        state_dict = aggregate_models(workers, admm_kwargs=admm_kwargs)

        for param, expected_param in zip(state_dict.values(), expected_params):
            npt.assert_almost_equal(param.tolist(), expected_param, decimal=3)


if __name__ == "__main__":
    unittest.main()
