import math
import random
import numpy as np
import gym
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
        critic = []
        for i, critic_layer in enumerate(hidden_units['critic'][1:]):
            if i > 0:
                critic.append(nn.ReLU())
            critic.append(nn.Linear(hidden_units['critic'][i], critic_layer))
        self.critic = nn.Sequential(*critic)

        hidden_units['actor'] = hidden_units['share'][-1:] + \
            hidden_units['actor'] + [num_outputs, ]
        actor = []
        for i, actor_layer in enumerate(hidden_units['actor'][1:]):
            if i > 0:
                actor.append(nn.ReLU())
            actor.append(nn.Linear(hidden_units['actor'][i], actor_layer))
        actor.append(nn.Softmax(dim=-1))
        self.actor = nn.Sequential(*actor)

    def forward(self, x):
        share = self.share(x)
        value = self.critic(share)
        prob = self.actor(share)
        return prob, value


class NormalAC(ActorCritic):
    pass


class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, data: tuple):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)


def compute_returns(next_value, rewards, masks, gamma=0.9):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns
