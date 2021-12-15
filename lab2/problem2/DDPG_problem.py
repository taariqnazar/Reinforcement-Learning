# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 2
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th November 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import trange
from DDPG_agent import Agent
import torch

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()

# Parameters
N_episodes = 300               # Number of episodes to run for training
discount_factor = 0.99         # Value of gamma
n_ep_running_average = 50      # Running average of 50 episodes
n_actions = len(env.action_space.high)   # dimensionality of the action
n_states = 8
buffer_size = 30000
target_network_update = 2
batch_size = 64
policy_update = 1e-3

mu=0.15
sigma=0.2
n = 0
# Reward
episode_reward_list = []  # Used to save episodes reward
episode_number_of_steps = []

# Agent initialization
agent = Agent(n_actions, n_states, batch_size, discount_factor, buffer_size, N_episodes, target_network_update, policy_update)

# Training process
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

for i in EPISODES:
    # Reset enviroment data
    done = False
    state = env.reset()
    total_episode_reward = 0
    t = 0
    while not done:
        if i % 50 == 0:
            # env.render()
            pass
        # Take a random action
        n = -mu*n + np.random.standard_normal(n_actions)
        action = agent.take_action(state) + n

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)
        agent.store(state, action, reward, next_state, done)
        agent.train()

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t+= 1

    # Append episode reward
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)
    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1]))


torch.save(agent.actor, 'neural-network-2-actor.pth')
torch.save(agent.critic, 'neural-network-2-critic.pth')
# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()
