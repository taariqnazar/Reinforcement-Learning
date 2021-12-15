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
# Last update: 26th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
from collections import deque, namedtuple
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from DDPG_soft_updates import soft_updates


class Actor(nn.Module):
    """
    Duelling Q network
    """
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, 400)
        self.linear2 = nn.Linear(400,200)
        self.linear3 = nn.Linear(200, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x =torch.tanh(self.linear3(x))
        return x

class Critic(nn.Module):
    """
    Duelling Q network
    """
    def __init__(self, input_size, n_actions):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, 400)
        self.linear2 = nn.Linear(400+n_actions,200)
        self.linear3 = nn.Linear(200, 1)

    def forward(self, x, u):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(torch.cat([x, u], 1)))
        x = self.linear3(x)
        return x


criterion = nn.MSELoss()
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class Agent(object):
    def __init__(self, n_actions, n_states, batch_size, discount_factor, buffer_size, n_episodes, target_network_update, policy_update, learning_rate_critic = 5e-4,learning_rate_actor = 5e-5):
        self.n_action = n_actions
        self.n_states = n_states
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.buffer_size = buffer_size
        self.n_episodes = n_episodes
        self.target_network_update = target_network_update
        self.policy_update = policy_update
        self.replay_buffer = deque([], maxlen=buffer_size)

        self.critic = self.__init_critic()
        self.critic_target = self.__init_critic()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate_critic)

        self.actor = self.__init_actor()
        self.actor_target = self.__init_actor()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate_actor)
        self.hard_update_target()  # Update target networks to parameters of og nets

        self.time = 0

    def __init_critic(self):
        return Critic(self.n_states, self.n_action)

    def __init_actor(self):
        return Actor(self.n_states, self.n_action)
    
    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.append(Experience(state, action, reward, next_state, done))
        self.time += 1

    def train(self):
        if self.time < self.batch_size:
            return

        experiences = random.sample(self.replay_buffer, self.batch_size - 1)
        experiences.append(self.replay_buffer[-1])
        batch = Experience(*zip(*experiences))

        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32)
        action_batch = torch.tensor(np.array(batch.action).reshape(-1, 2), dtype=torch.float32)
        reward_batch = torch.tensor(np.array(batch.reward), dtype=torch.float32)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32)
        none_final_batch = torch.tensor(~np.array(batch.done))

        state_action_values = self.critic(state_batch, action_batch)

        next_state_values = torch.zeros(self.batch_size)
        next_state_values[none_final_batch] = self.critic_target(next_state_batch[none_final_batch], self.actor_target(next_state_batch[none_final_batch])).squeeze()

        expected_return = self.discount_factor*next_state_values + reward_batch
        loss = criterion(state_action_values, expected_return.unsqueeze(1))

        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        if self.time % self.target_network_update:
            self.actor.zero_grad()

            policy_loss = -self.critic(state_batch, self.actor(state_batch))
            policy_loss  = policy_loss.mean()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
            self.actor_optimizer.step()

            self.actor_target = soft_updates(self.actor, self.actor_target, self.policy_update)
            self.critic_target = soft_updates(self.critic, self.critic_target, self.policy_update)


    def take_action(self, state):
        with torch.no_grad():
            return self.actor(torch.tensor(state)).numpy()

    def hard_update_target(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())



class RandomAgent(object):
    ''' Agent taking actions uniformly at random, child of the class Agent'''

    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def take_action(self, state: np.ndarray, episode) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent.
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)




    
