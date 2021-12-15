import gym
import numpy as np
import torch

env = gym.make('LunarLander-v2')
env.reset()

# Parameters
N_episodes = 50                             # Number of episodes

# Random agent initialization
model = torch.load('neural-network-1.pth')

rewards_list = []
for i in range(N_episodes):
    # Reset enviroment data and initialize variables
    done = False
    state = env.reset()

    total_episode_reward = 0
    while not done:
        env.render()

        q_values = model(torch.tensor([state]))
        _, action = torch.max(q_values, axis=1)
        next_state, reward, done, _ = env.step(action.item())

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state

    state = next_state

    rewards_list.append(total_episode_reward)
    print(f'Episode: {i + 1}, Total Reward: {total_episode_reward}')
    # Close environment
    env.close()

print(f'Confidence Interval of 95%: {np.mean(rewards_list)} +- {1.96*(np.std(rewards_list)/np.sqrt(len(rewards_list)))}')

