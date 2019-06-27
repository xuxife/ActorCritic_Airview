import torch
import torch.optim as optim
from torch.distributions import Categorical


from simulator import *
from actor_critic import *
from dqn import *

import matplotlib.pyplot as plt
from concurrent import futures
import time



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
                print("%.2f" % average_rewards[-1])

        if replay is not None:
            if len(replay) > replay_size:
                actor_loss, critic_loss, entropy = replay_loss_AC(
                    model, *replay.sample(replay_size))
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
                action = torch.randint(1,value.shape[1],(value.shape[0],)) 

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
                print("%.2f" % average_rewards[-1])

            if frame_idx % 10000 == 0 and frame_idx > 30000:
                print('saving model')
                torch.save(model.state_dict(), "DQN_model.pt")

        if replay is not None:
            if len(replay) == replay.capacity:
                loss = replay_loss_DQN(model, *replay.sample(replay_size))
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
        next_state, reward, done, info = env.step(np.clip(action+alter,1,29))
        total_reward += reward
        total_deliver += info['total']
        success_rate.append(total_reward/total_deliver)
        frame_idx += 1
        average_rewards.append(total_reward/frame_idx)
        if frame_idx % 1000 == 0:
            print("%.2f" % average_rewards[-1])
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
            print("%.2f" % average_rewards[-1])
        if done:
            env.reset()
    return average_rewards, success_rate


def replay_loss_AC(model, state, action, reward, done, next_state):
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


def replay_loss_DQN(model, batch_state, batch_action, batch_reward, batch_done, batch_next_state):
    loss = 0
    batch_reward = torch.FloatTensor(np.vstack(batch_reward))

    for state, action, reward in zip(batch_state,batch_action,batch_reward):
        value = model(state)
        eval_ = torch.sum(value.gather(1,action.unsqueeze(dim=1)))
        loss += (eval_-reward)**2

    return loss


def plot_fig(rewards, title, start_step=5000, save=False):
    plt.figure()
    for i,average_rewards in enumerate(rewards):
        plt.plot(average_rewards[start_step:],label=f"{i}th run")
    plt.title(title,fontsize=15)
    plt.xlabel("Steps",fontsize=10)
    plt.ylabel("Average_rewards",fontsize=10)
    plt.legend(loc="best")
    plt.show()
    if save: plt.savefig(f"{title}.png")


def run_model(train_fun,inputs):
    """
    use multi-threading to run train_fun, given inputs
    Args:
        train_fun: the training function. i.e. trainDQN/trainAC/random_test etc.
        inputs: list(args of train_fun)

    Returns:
        eg.
            rewards(list(rewards)): model rewards of different inputs
            success_rates(list(success_rates)): model success_rates of different inputs
            
            the return format of rewards and sucess_rate are as follows:
            [inputs1,inputs2,...]

    Examples:
        inputs = [(env1,model1,opt1), (env2,model2,opt2),...]
        rewards, sucess_rate = run_model(trainAC,inputs)
        plot_fig(rewards) 
    """
    rewards = []
    success_rates = []

    with futures.ThreadPoolExecutor(max_workers=30) as executor:
        future_list = []
        for input_ in inputs:
            future = executor.submit(train_fun,*input_)
            future_list.append(future)
            
        for future in futures.as_completed(future_list):
            rewards.append(future.result()[0])
            success_rates.append(future.result()[1])

    return rewards, success_rates

time.sleep(2000)
# train experiment
for i in range(20):
    env = Airview(episode_length=10, ue_arrival_rate=0.05)
    state_dim = env.observation_space.shape[1]
    action_dim = 29
    net_hidden = [126,64,32,16]


    max_frames = 500000
    num_steps = 5
    epsilon = 0.9
    replay_size = 20
    replay_buffer_size = 2000
    replay = ReplayBuffer(replay_buffer_size)

    default_para = (max_frames,num_steps,epsilon,replay,replay_size)

    inputs = []
    all_rewards = []

    # DQN
    model = DQN(state_dim,action_dim)
    opt = torch.optim.Adam(model.parameters())
    env = Airview(episode_length=10, ue_arrival_rate=0.05)
    replay = ReplayBuffer(replay_buffer_size)
    inputs = [(env,model,opt,*default_para)]
    rewards,success_rate = run_model(trainDQN,inputs)
    all_rewards.append(rewards[0])

    # AC
    model = ActorCritic(state_dim,action_dim)
    opt = torch.optim.Adam(model.parameters())
    env = Airview(episode_length=10, ue_arrival_rate=0.05)
    replay = ReplayBuffer(replay_buffer_size)
    inputs = [(env,model,opt,max_frames,num_steps,replay,replay_size)]
    rewards,success_rate = run_model(trainAC,inputs)
    all_rewards.append(rewards[0])

    # AVG
    model = Policy(mode="avg")
    env = Airview(episode_length=10, ue_arrival_rate=0.05)
    rewards,success_rate = baseline(env,model,max_frames=max_frames)
    all_rewards.append(rewards)

    # AVG-3
    model = Policy(mode="avg")
    env = Airview(episode_length=10, ue_arrival_rate=0.05)
    rewards,success_rate = baseline(env,model,max_frames=max_frames,alter=-3)
    all_rewards.append(rewards)

    # SNR-3
    model = Policy(mode="snr")
    env = Airview(episode_length=10, ue_arrival_rate=0.05)
    rewards,success_rate = baseline(env,model,max_frames=max_frames,alter=-3)
    all_rewards.append(rewards)


    model_names = ["DQN","AC","AVG","AVG-3","SNR-3"]
    plt.figure()
    for rewards, name in zip(all_rewards,model_names):
        plt.plot(rewards[5000:],label=name)
    plt.xlabel('step')
    plt.ylabel('average reward')
    plt.legend(loc='best')
    plt.savefig(f"All_Comparation_{i}.png")
    plt.show()




# ----------   experiment on baseline    --------------------
# alters = list(range(-3,4))
# avg_rewards = []
# model = Policy()

# for alter in alters:
#     print(f"alter:{alter}")
#     average_rewards, _ = baseline(env,model,max_frames=200000,alter=alter)
#     avg_rewards.append(average_rewards)

# plt.figure()

# for (alter, average_rewards) in zip(alters, avg_rewards):
#     plt.plot(average_rewards[3000:],label=f'alter:{alter}')

# plt.xlabel('step')
# plt.ylabel('average reward')
# plt.title('SNR')
# plt.legend(loc='best')
# plt.show()