import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_units={'share': [128, ], 'critic': [32, ], 'actor': [64, ]}):
        super().__init__()
        hidden_units['share'] = [num_inputs, ] + hidden_units['share']
        share = []
        for i, share_layer in enumerate(hidden_units['share'][1:]):
            share.append(nn.Linear(hidden_units['share'][i], share_layer))
            share.append(nn.ReLU())
        self.share = nn.Sequential(*share)

        hidden_units['critic'] = hidden_units['share'][-1:] + \
            hidden_units['critic'] + [1, ]
        critic = [self.share, ]
        for i, critic_layer in enumerate(hidden_units['critic'][1:]):
            if i > 0:
                critic.append(nn.ReLU())
            critic.append(nn.Linear(hidden_units['critic'][i], critic_layer))
        self.critic = nn.Sequential(*critic)

        hidden_units['actor'] = hidden_units['share'][-1:] + \
            hidden_units['share'] + [num_outputs, ]
        actor = [self.share, ]
        for i, actor_layer in enumerate(hidden_units['actor'][1:]):
            if i > 0:
                actor.append(nn.ReLU())
            actor.append(nn.Linear(hidden_units['actor'][i], actor_layer))
        self.actor = nn.Sequential(*actor)

    def forward(self, x):
        value = self.critic(x)
        prob = self.actor(x)
        return prob, value


def compute_returns(next_value, rewards, masks, gamma=0.9):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns
