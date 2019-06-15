import torch
import torch.optim as optim
from torch.distributions import Categorical

from simulator import *
# from simulator_rl import *
from actor_critic import *

import matplotlib.pyplot as plt

env = Airview(episode_length=10)
# env = Airview(ue_arrival_rate=0.5, episode_tti=10)

state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
action_dim = env.action_space.shape[0] * 29
# state_dim = env.state_dim
# action_dim = env.action_dim

model = ActorCritic(state_dim, action_dim, 256)
# with open("model.pkl", 'rb') as f:
# model = torch.load(f)
optimizer = optim.Adam(model.parameters())


def train(env, max_frames=50000, num_steps=5, gamma=0.9):
    frame_idx = 0
    total_reward = 0
    average_rewards = []

    success_deliver = 0
    total_deliver = 0
    success_rate = []

    state = env.reset()

    while frame_idx < max_frames:
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0

        for _ in range(num_steps):
            state = torch.FloatTensor(state.flatten())
            prob, value = model(state)
            prob = prob.view(17, -1).softmax(dim=-1)
            dist = Categorical(prob)
            action = dist.sample()

            next_state, reward, done, info = env.step(action)
            # next_state, reward, done, _, _, _, action = env.step(
            # prob.view(action_dim).detach().numpy())
            success_deliver += info['success']
            total_deliver += 17
            success_rate.append(success_deliver/total_deliver)

            total_reward += reward
            log_prob = dist.log_prob(
                torch.LongTensor(action)).sum().unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor([reward]).unsqueeze(1))
            masks.append(torch.FloatTensor([1-done]).unsqueeze(1))

            state = next_state
            frame_idx += 1

            average_rewards.append(total_reward / frame_idx)
            if frame_idx % 1000 == 0:
                print(average_rewards[-1])

        next_state = torch.FloatTensor(next_state)
        _, next_value = model(next_state.flatten())
        returns = compute_returns(next_value, rewards, masks, gamma=gamma)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        # loss = actor_loss + 0.5*critic_loss - 0.001 * entropy
        loss = actor_loss + 0.5*critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if done:
            # total_reward = 0
            state = env.reset()
    return average_rewards, success_rate


def baseline(env, policy, max_frames=10000, alter=0):
    frame_idx = 0
    total_reward = 0
    average_rewards = []
    success_deliver = 0
    total_deliver = 0
    success_rate = []
    state = env.reset()
    while frame_idx < max_frames:
        action = policy.decide(env.select_ue)
        next_state, reward, done, info = env.step(action)
        success_deliver += info['success']
        total_deliver += 17
        success_rate.append(success_deliver/total_deliver)
        total_reward += reward
        frame_idx += 1
        average_rewards.append(total_reward/frame_idx)
        if frame_idx % 1000 == 0:
            print(average_rewards[-1])
        if done:
            env.reset()
    return average_rewards, success_rate


def random_test(env, max_frames=10000):
    frame_idx = 0
    total_reward = 0
    average_rewards = []
    success_deliver = 0
    total_deliver = 0
    success_rate = []

    state = env.reset()
    while frame_idx < max_frames:
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        success_deliver += info['success']
        total_deliver += 17
        success_rate.append(success_deliver/total_deliver)

        total_reward += reward
        frame_idx += 1
        average_rewards.append(total_reward/frame_idx)
        if frame_idx % 1000 == 0:
            print(average_rewards[-1])
        if done:
            env.reset()
    return average_rewards, success_rate
