import numpy as np
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_units=[128, 64]):
        super().__init__()
        units = [num_inputs, ] + hidden_units + [num_outputs, ]
        layers = []
        for i, layer_unit in enumerate(units[1:]):
            layers.append(nn.Linear(units[i], layer_unit))
            if i < len(units)-2:
                layers.append(nn.ReLU())
        self.value = nn.Sequential(*layers)

    def forward(self, x):
        return self.value(x)
