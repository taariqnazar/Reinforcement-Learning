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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    """
    Duelling Q network
    """

    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.linear_1 = nn.Linear(input_size, 64)
        self.linear_2 = nn.Linear(64, 64)

        self.linear_h_v = nn.Linear(64,32)
        self.linear_z_v = nn.Linear(32,1)

        self.linear_h_a = nn.Linear(64,32)
        self.linear_z_a = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))

        v = F.relu(self.linear_h_v(x))
        v = self.linear_z_v(v)

        a = F.relu(self.linear_h_a(x))
        a = self.linear_z_a(a)

        q = v + a - a.mean()
        return q

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))


class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''

    def __init__(self, n_episodes, n_actions, n_states, batch_size, epsilon_min, epsilon_max, buffer_size, discount_factor, learning_rate=1e-4):
        self.n_episodes = n_episodes
        self.n_actions = n_actions
        self.n_states = n_states
        self.batch_size = batch_size
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.buffer_size = buffer_size
        self.discount_factor = discount_factor
        self.network_update_frequency = int(buffer_size/batch_size)

        self.replay_buffer = deque([], maxlen=buffer_size)

        self.network = self.__init_network()
        self.target_network = self.__init_network()
        self.update_target()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        self.last_action = None
        self.time = 0

    def __init_network(self):
        network = Net(self.n_states, self.n_actions)
        return network

    def load_model(self, pth):
        self.network = torch.load(pth)

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.append(Experience(state, action, reward, next_state, done))
        self.time += 1

    def train_network(self):
        if self.time < self.batch_size:
            return

        experiences = random.sample(self.replay_buffer, self.batch_size - 1)
        experiences.append(self.replay_buffer[-1])

        batch = Experience(*zip(*experiences))

        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32)
        action_batch = torch.tensor(np.array(batch.action).reshape(-1, 1))
        reward_batch = torch.tensor(np.array(batch.reward), dtype=torch.float32)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32)
        none_final_batch = torch.tensor(~np.array(batch.done))

        state_action_values = self.network(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size)
        next_state_values[none_final_batch] = self.target_network(next_state_batch[none_final_batch]).max(1)[0].detach()

        expected_return = self.discount_factor*next_state_values + reward_batch
        loss = self.criterion(state_action_values, expected_return.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.15)
        self.optimizer.step()

        if self.time % self.network_update_frequency:
            self.update_target()

    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def take_action(self, state, episode):
        epsilon = np.max([self.epsilon_min, self.epsilon_max*((self.epsilon_min/self.epsilon_max)**((episode)/(int(self.n_episodes*0.9)-1)))])
        # epsilon = np.max([self.epsilon_min, self.epsilon_max - (((self.epsilon_max - self.epsilon_min)*(episode))/((int(self.n_episodes*0.9)-1)))])
        if np.random.uniform() < epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            with torch.no_grad():
                return np.argmax(self.network(torch.from_numpy(state)).numpy())


