from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        out = self.embedding(x)
        out, hidden = self.gru(out, hidden)
        out = self.linear(out)
        return F.log_softmax(out, dim=2).transpose(1, 2), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return weight.new_zeros(1, batch_size, self.hidden_size)
