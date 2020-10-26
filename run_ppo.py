import random

import gym
import envs
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import matplotlib.pyplot as plt

from common.multiprocessing_env import SubprocVecEnv

from models import blackbox_synthetic_GIN, ActorCritic

num_envs = 4
env_name = "CustomEnv-v0"
model_file = "save/tree_grid_star_model.pkl"

num_features = 3
num_classes = 4

max_nodes = 15
min_nodes = 5

ckpt = torch.load(model_file)
blackbox_model = blackbox_synthetic_GIN(num_features, num_classes)
blackbox_model.load_state_dict(ckpt["model"])
blackbox_model.eval()

def make_env(num_features, blackbox_model, c, max_nodes, min_nodes):
    def _thunk():
        env = gym.make(env_name, num_features=num_features, 
        	blackbox_model=blackbox_model, c=c, max_nodes=max_nodes,
            min_nodes=min_nodes)
        return env

    return _thunk

    
def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        
        

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):

            dist, action, value = model(state)

            entropy = torch.mean(torch.stack([dist[0].entropy().mean(),
                                            dist[1].entropy().mean(),
                                            dist[2].entropy().mean()]))

            #entropy = dist.entropy().mean()
            #new_log_probs = dist.log_prob(action)

            #Take mean of log probs and entropy across action components
            dist_A_log_prob = dist[0].log_prob(action[:, 0])
            dist_B_log_prob = dist[1].log_prob(action[:, 1])
            dist_C_log_prob = dist[2].log_prob(action[:, 2])

            log_prob = [dist_A_log_prob, dist_B_log_prob, dist_C_log_prob]
            log_prob = torch.mean(torch.stack(log_prob), 0)
            new_log_probs = torch.reshape(log_prob, (-1, 1))

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


#Hyper params:
embedding_size   = 64 #size of node embeddings
lr               = 3e-4
num_steps        = 20 #20
mini_batch_size  = 5 #5
ppo_epochs       = 4
threshold_reward = -200

for c in range(num_classes):

    print("Learning Policy for class:", c)

    envs = [make_env(num_features, blackbox_model, c, max_nodes, min_nodes) for i in range(num_envs)]
    envs = SubprocVecEnv(envs)

    model = ActorCritic(num_features, embedding_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    max_frames = 1500
    frame_idx  = 0
    test_rewards = []

    state = envs.reset()

    early_stop = False

    #Save mean rewards per episode
    env_0_mean_rewards = []
    #env_1_mean_rewards = []

    while frame_idx < max_frames and not early_stop:

        log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        masks     = []
        entropy = 0

        env_0_rewards = []

        for _step in range(num_steps):

            state = torch.FloatTensor(state)
            dist, action, value = model(state)

            dist_A = dist[0]
            dist_B = dist[1]
            dist_C = dist[2]


            #Environment is automatically reset when done is True
            next_state, reward, done, _ = envs.step(action.cpu().numpy())

            env_0_rewards.append(reward[0])
            #env_1_rewards.append(reward[1])

            if(done[0]):
                env_0_mean_rewards.append(np.mean(env_0_rewards))
                env_0_rewards = []

                #print(state[0])


            #Take mean of log probs and entropy across action components
            dist_A_log_prob = dist_A.log_prob(action[:, 0])
            dist_B_log_prob = dist_B.log_prob(action[:, 1])
            dist_C_log_prob = dist_C.log_prob(action[:, 2])

            log_prob = [dist_A_log_prob, dist_B_log_prob, dist_C_log_prob]
            log_prob = torch.mean(torch.stack(log_prob), 0)
            log_prob = torch.reshape(log_prob, (-1, 1))


            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1))


            states.append(state)
            actions.append(action)

            state = next_state
            frame_idx += 1

            if(frame_idx % 100 == 0):

                print("Frame Index:", frame_idx)
            #if frame_idx % 1000 == 0:
            #    test_reward = np.mean([test_env() for _ in range(10)])
            #    test_rewards.append(test_reward)
            #    plot(frame_idx, test_rewards)
            #    if test_reward > threshold_reward: early_stop = True
                


        next_state = torch.FloatTensor(next_state)
        _, _, next_value = model(next_state)
        returns = compute_gae(next_value, rewards, masks, values)

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        advantage = returns - values



        ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)


    #Plot training rewards for environments 0 and 1
    episodes = [i for i in range(len(env_0_mean_rewards))]
    plt.plot(episodes, env_0_mean_rewards)
    #plt.plot(steps, env_1_rewards)
    plt.xlabel("episodes")
    plt.ylabel("Reward")
    filename = 'PPO_Class_' + str(c) + '_Rewards.png'
    plt.savefig(filename)
    plt.clf()







#env = gym.make(env_name)


















"""

from itertools import count

max_expert_num = 50000
num_steps = 0
expert_traj = []

for i_episode in count():
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        dist, _ = model(state)
        action = dist.sample().cpu().numpy()[0]
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        expert_traj.append(np.hstack([state, action]))
        num_steps += 1
    
    print("episode:", i_episode, "reward:", total_reward)
    
    if num_steps >= max_expert_num:
        break
        
expert_traj = np.stack(expert_traj)
print()
print(expert_traj.shape)
print()
np.save("expert_traj.npy", expert_traj)

"""