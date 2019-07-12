from __future__ import absolute_import, division, print_function

import collections
import functools

import datasets.femnist as femnist
import datasets.shakespeare as shakespeare
from nets.lenet import LeNet
from nets.rnn import RNN
from nets.cnn import CNN
from utils.train import train_model, train_rnn


NetArguments = collections.namedtuple(
    'NetArguments',
    [
        'model',
        'dataset',
        'partition_kwargs',
        'train_fn',
    ]
)


def create_net(model_name, **kwargs):
    model_name = model_name.lower()
    partition_kwargs = {}
    train_fn = train_model
    if model_name == 'lenet':
        model = LeNet()
        dataset = femnist
    elif model_name == 'cnn':
        model = CNN()
        dataset = femnist
    elif model_name == 'rnn':
        dataset = shakespeare
        model = RNN(
            vocab_size=len(shakespeare.ShakespeareDataset._VOCAB),
            embedding_dim=kwargs.get('embedding_dim', 100),
            hidden_size=kwargs.get('hidden_size', 128))
        partition_kwargs = {'seq_length': kwargs.get('seq_length', 50)}
        hidden = model.init_hidden(kwargs.get('batch_size', 1))
        train_fn = functools.partial(train_rnn, hidden=hidden)
    else:
        raise ValueError('Unknown model: ' + model_name)

    return NetArguments(model=model,
                        dataset=dataset,
                        partition_kwargs=partition_kwargs,
                        train_fn=train_fn)
