from __future__ import absolute_import, division, print_function

import unittest
from unittest.mock import Mock
import torch.nn as nn
import numpy.testing as npt

from sequential.worker import *


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
            'tolerance': 1e-3,
            'lr': 0.01,
        }

        model1 = create_test_model(weight1, bias1)
        model2 = create_test_model(weight2, bias2)

        workers = [
            Mock(model=model1),
            Mock(model=model2),
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
            'tolerance': 1e-3,
            'lr': 0.01,
        }

        model1 = create_test_model(weight1, bias1)
        model2 = create_test_model(weight2, bias2)

        workers = [
            Mock(model=model1),
            Mock(model=model2),
        ]

        expected_params = [expected_weight, expected_bias]
        state_dict = aggregate_models(
            workers, weights=aggr_weights, admm_kwargs=admm_kwargs)

        for param, expected_param in zip(state_dict.values(), expected_params):
            npt.assert_almost_equal(param.tolist(), expected_param, decimal=3)


if __name__ == "__main__":
    unittest.main()
