import numpy as np
import gym

#TODO:
#Create agent
#Create Environment
#Train Agent
if __name__ == '__main__':
   n_episodes = 5
   n_steps = 200
   
   env = gym.make('LunarLander-v2')
   for episode in range(10):
      
      state = env.reset()
      done = False
      for t in range(n_steps):
         env.render()
         next_state, reward, done, _ = env.step(env.action_space.sample())
         if done:
            print(f'Episode finished after {t+1} timesteps')
            break
   env.close()