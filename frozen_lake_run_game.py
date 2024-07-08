# https://www.youtube.com/watch?v=QK_PP_2KgGE&list=PLIfai4buiQei1-BfmVjUxN7CpNgqNJn2Z
import numpy as np
import gym
import random
import time

# tunable parameters
num_episodes = 3
max_steps_per_episode = 100

env = gym.make('FrozenLake-v1', render_mode = 'human')
exploration_rate = 1
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.loadtxt(fname='data/frozen_lake_q_table.csv', delimiter=',')

rewards_all_episodes = []

# Q-Learning Algo
for episode in range(1,num_episodes + 1):
    state = env.reset()[0]

    done = False
    rewards_current_episode = 0
    print(f'\r\n\n*********Episode {episode}\n\n\n')

    time.sleep(1)
    
    for step in range(max_steps_per_episode):

        env.render()
        time.sleep(0.3)

        #Exploration-Exploitation 
        action = np.argmax(q_table[state, :])
        
        new_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        if done:
            env.render()
            outcome = 'you fell in the ice'
            if reward > 0:
                outcome = 'you won'
            print(f'on episode {episode}, {outcome} after {step} steps')
            break

        
        
        state = new_state

env.close()
