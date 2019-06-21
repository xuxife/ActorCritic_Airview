import torch
import torch.optim as optim
from torch.distributions import Categorical


from simulator import *
from actor_critic import *
from dqn import *

import matplotlib.pyplot as plt

TEST_TIME = 5

env = Airview(episode_length=10, ue_arrival_rate=0.1)
state_dim = env.observation_space.shape[1]
action_dim = 29

model = ActorCritic(state_dim, action_dim, {
                    'share': [128, ], 'critic': [32, ], 'actor': [64, ]})
model = DQN(state_dim, action_dim, [128, 64])
# with open("model.pkl", 'rb') as f:
# model = torch.load(f)
optimizer = optim.Adam(model.parameters())
buffer = ReplayBuffer(10000)


def trainAC(env, model, optimizer, max_frames=50000, num_steps=5, replay=None, replay_size=20):
    frame_idx = 0
    total_reward = 0
    average_rewards = []
    total_deliver = 0
    success_rate = []
    state = env.reset()
    state = torch.FloatTensor(state)
    while frame_idx < max_frames:
        log_probs = []
        values = []
        rewards = []
        masks = []
        actor_loss = 0
        critic_loss = 0
        entropy = 0
        for _ in range(num_steps):
            frame_idx += 1
            prob, value = model(state)
            # prob = prob.view(17, -1).softmax(dim=-1)
            dist = Categorical(prob)
            action = dist.sample()

            next_state, reward, done, info = env.step(action)
            next_state = torch.FloatTensor(next_state)
            total_reward += reward
            average_rewards.append(total_reward / frame_idx)
            total_deliver += info['total']
            success_rate.append(total_reward/total_deliver if total_deliver > 0 else 0)
            if replay is not None:
                replay.push((state, action, reward, done, next_state))
            else:
                log_prob = dist.log_prob(action).sum().unsqueeze(0)
                entropy += dist.entropy().mean()
                advantage = reward - value
                actor_loss += -(log_prob*advantage.detach()).mean()
                critic_loss += advantage.pow(2).mean()

            state = next_state
            if frame_idx % 1000 == 0:
                print(average_rewards[-1])

        if replay is not None:
            if len(replay) > replay_size:
                actor_loss, critic_loss, entropy = replay_loss(
                    model, optimizer, *replay.sample(replay_size))
            else:
                continue

        loss = actor_loss + 0.5*critic_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if done:
            state = env.reset()
            state = torch.FloatTensor(state)
    return average_rewards, success_rate


def trainDQN(env, model, optimizer, max_frames=50000, num_steps=5, epsilon=0.9, replay=None, replay_size=20):
    frame_idx = 0
    total_reward = 0
    average_rewards = []
    total_deliver = 0
    success_rate = []
    state = env.reset()
    state = torch.FloatTensor(state)
    plt.ion()
    while frame_idx < max_frames:
        for _ in range(num_steps):
            frame_idx += 1
            value = model(state)
            if np.random.uniform() < epsilon:
                action = value.argmax(dim=-1)
                
            else:
                action = torch.randint(1,value.shape[1],(value.shape[0],)) #TO-TEST

            next_state, reward, done, info = env.step(action)
            next_state = torch.FloatTensor(next_state)
            total_reward += reward
            average_rewards.append(total_reward / frame_idx)
            total_deliver += info['total']
            success_rate.append(total_reward/total_deliver if total_deliver > 0 else 0)
            if replay is not None:
                replay.push((state, action, reward, done, next_state))
            else:
                pass # TODO

            state = next_state
            if frame_idx % 1000 == 0:
                print(average_rewards[-1])
                # plt.clf()
                # plt.plot(average_rewards[3000:])
                # plt.pause(1.5) # for plotting the figure

            if frame_idx % 10000 == 0 and frame_idx > 30000:
                print('saving model')
                torch.save(model.state_dict(), "DQN_model.pt")

        if replay is not None:
            if len(replay) == replay.capacity:
                batch = list(replay.sample(replay_size))

                batch_state = batch[0]
                batch_action = batch[1]
                batch_reward = torch.FloatTensor(np.vstack(batch[2]))
                loss_func = nn.MSELoss()
                loss = 0

                for state, action,reward in zip(batch_state,batch_action,batch_reward):
                    value = model(state)
                    eval_ = torch.sum(value.gather(1,action.unsqueeze(dim=1)))
                    loss += torch.abs(eval_-reward)**2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            else:
                continue


        if done:
            state = env.reset()
            state = torch.FloatTensor(state)
    return average_rewards, success_rate


def baseline(env, policy, max_frames=10000, alter=0):
    frame_idx = 0
    total_reward = 0
    average_rewards = []
    total_deliver = 0
    success_rate = []
    state = env.reset()
    while frame_idx < max_frames:
        action = policy.decide(env.sched_ue_count.keys())
        next_state, reward, done, info = env.step(action+alter)
        total_reward += reward
        total_deliver += info['total']
        success_rate.append(total_reward/total_deliver)
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
        total_reward += reward
        total_deliver += info['total']
        success_rate.append(total_reward/total_deliver)
        frame_idx += 1
        average_rewards.append(total_reward/frame_idx)
        if frame_idx % 1000 == 0:
            print(average_rewards[-1])
        if done:
            env.reset()
    return average_rewards, success_rate


def replay_loss(model, optimizer, state, action, reward, done, next_state):
    actor_loss = 0
    critic_loss = 0
    entropy = 0
    for i in range(len(state)):
        prob, value = model(torch.FloatTensor(state[i]))
        dist = Categorical(prob)
        log_prob = dist.log_prob(action[i]).sum().unsqueeze(0)
        entropy += dist.entropy().mean()
        advantage = torch.FloatTensor([reward[i], ]) - value
        actor_loss += -(log_prob*advantage.detach()).mean()
        critic_loss += advantage.pow(2).mean()
    return actor_loss, critic_loss, entropy

# train experiment
env = Airview(ue_arrival_rate=0.05)
# model = ActorCritic(5,29)
model = DQN(5,29)
opt = torch.optim.Adam(model.parameters())
replay = ReplayBuffer(1000)

avg_rewards = []
for i in range(1):
    average_rewards, success_rate = trainDQN(env,model,opt,replay=replay,max_frames=200000,replay_size=128)
    avg_rewards.append(average_rewards)

plt.figure()
for average_rewards in avg_rewards:
    plt.plot(average_rewards[5000:])
plt.title("Actor Critic",fontsize=15)
plt.xlabel("Steps",fontsize=10)
plt.ylabel("Average_rewards",fontsize=10)
plt.savefig("AC_performance.png")
plt.show()








# ----------   experiment on baseline    --------------------
# alters = list(range(-3,4))
# avg_rewards = []
# model = Policy()

# for alter in alters:
#     print(f"alter:{alter}")
#     average_rewards, _ = baseline(env,model,max_frames=200000,alter=10)
#     avg_rewards.append(average_rewards)

# plt.figure()

# for (alter, average_rewards) in zip(alters, avg_rewards):
#     plt.plot(average_rewards[3000:],label=f'alter:{alter}')

# plt.xlabel('step')
# plt.ylabel('average reward')
# plt.title('baseline')
# plt.legend(loc='best')
# plt.show()