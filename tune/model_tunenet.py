#!/usr/bin/env python
#

#
# TuneNet: a 2-input neural network for estimating model
# parameter differences.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data

from tune.utils import get_torch_device


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.fc = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        output = F.relu(output)
        output = self.out(output)
        return output, None


class TuneNet(nn.Module):
    hidden_size = 32

    def __init__(self,
                 input_size,
                 output_size,
                 input_size2=None,
                 output_fn=torch.tanh,
                 degenerate=False):
        super(TuneNet, self).__init__()

        self.input_size = input_size
        if input_size2 is None:
            self.input_size2 = input_size
        else:
            self.input_size2 = input_size2
        self.output_fn = output_fn
        self.degenerate = degenerate
        self.device = get_torch_device()

        degenerate_string = ""
        if self.degenerate:
            degenerate_string = "[Degenerate] "
        print("Creating {}TuneNet with input sizes {} and {}, hidden size {}".format(
            degenerate_string,
            self.input_size,
            self.input_size2,
            self.hidden_size))

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.input_size2, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)

    def forward(self, inp):
        inp1 = inp[:, :self.input_size]
        if self.degenerate:
            inp1 = torch.zeros(inp1.shape).to(self.device)
        o1 = F.relu(self.fc1(inp1))
        inp2 = inp[:, self.input_size:]
        o2 = F.relu(self.fc2(inp2))
        # print(torch.sub(o1, o2))
        # print(inp.shape)
        # print(o1.shape)
        # print(torch.sub(o1, o2).shape)
        o3 = F.relu(self.fc3(torch.cat([o1, o2], 1)))
        output = self.output_fn(self.out(o3))
        return output
