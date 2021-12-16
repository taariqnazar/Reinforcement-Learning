import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from DQN_agent import RandomAgent

env = gym.make('LunarLander-v2')
env.reset()

model = torch.load('neural-network-1.pth')
random_agent = RandomAgent(4)

# Parameters
N_EPISODES = 50            # Number of episodes to run for trainings
CONFIDENCE_PASS = 50

# Reward
episode_reward_list = []  # Used to store episodes reward

# Simulate episodes
for i in range(N_EPISODES):
    # Reset enviroment data
    done = False
    state = env.reset()
    total_episode_reward = 0.
    while not done:
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        q_values = model(torch.tensor([state]))
        _, action = torch.max(q_values, axis=1)
        next_state, reward, done, _ = env.step(action.item())

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state

    # Append episode reward
    episode_reward_list.append(total_episode_reward)

    # Close environment
    env.close()


episode_reward_list_rand = []  # Used to store episodes reward

for i in range(N_EPISODES):
    # Reset enviroment data
    done = False
    state = env.reset()
    episode_reward_rand = 0.
    while not done:
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(np.random.randint(4))

        # Update episode reward
        episode_reward_rand += reward

        # Update state for next iteration
        state = next_state

    # Append episode reward
    episode_reward_list_rand.append(episode_reward_rand)

    # Close environment
    env.close()

plt.plot([i for i in range(N_EPISODES)], episode_reward_list)
plt.plot([i for i in range(N_EPISODES)], episode_reward_list_rand)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend(['Agent', 'Random Agent'])
plt.show()
