from __future__ import absolute_import, division, print_function

import collections
import functools

import nets.lenet as lenet
import nets.rnn as rnn
import nets.cnn as cnn
import nets.svm as svm
from utils.train import train_model, train_rnn


NetArguments = collections.namedtuple(
    'NetArguments',
    [
        'model',
        'load_dataset_fn',
        'train_fn',
        'test_fn',
        'loss_fn',
    ]
)


def create_net(model_name, **kwargs):
    model_name = model_name.lower()
    train_fn = train_model
    if model_name == 'lenet':
        load_dataset_fn = lenet.dataset.load_dataset
        loss_fn = lenet.loss_fn
        test_fn = lenet.test_fn
        model = lenet.LeNet()
    elif model_name == 'cnn':
        load_dataset_fn = cnn.dataset.load_dataset
        loss_fn = cnn.loss_fn
        test_fn = cnn.test_fn
        model = cnn.CNN()
    elif model_name == 'rnn':
        load_dataset_fn = functools.partial(
            rnn.dataset.load_dataset, seq_length=kwargs.get('seq_length', 50))
        loss_fn = rnn.loss_fn
        test_fn = rnn.test_fn
        model = rnn.RNN(
            vocab_size=len(rnn.dataset.VOCAB),
            embedding_dim=kwargs.get('embedding_dim', 100),
            hidden_size=kwargs.get('hidden_size', 128))
        hidden = model.init_hidden(kwargs.get('batch_size', 1))
        train_fn = functools.partial(train_rnn, hidden=hidden)
    elif model_name == 'svm':
        load_dataset_fn = svm.dataset.load_dataset
        loss_fn = svm.loss_fn
        test_fn = svm.test_fn
        model = svm.LinearSVM()
    else:
        raise ValueError('Unknown model: ' + model_name)

    return NetArguments(model=model,
                        load_dataset_fn=load_dataset_fn,
                        train_fn=train_fn,
                        test_fn=test_fn,
                        loss_fn=loss_fn)
