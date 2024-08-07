# https://www.youtube.com/watch?v=QK_PP_2KgGE&list=PLIfai4buiQei1-BfmVjUxN7CpNgqNJn2Z
import numpy as np
import gym
import random
import time

env = gym.make('FrozenLake-v1')

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))
# print(q_table)

# tunable parameters
num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []

# Q-Learning Algo
for episode in range(num_episodes):
    state = env.reset()[0]

    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):

        #Exploration-Exploitation 
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()
        
        new_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated


        if reward > 0:
            apple = 1

        # update q-table
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state,:]))
        
        state = new_state
        rewards_current_episode += reward

        if done:
            break
    
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    rewards_all_episodes.append(rewards_current_episode)

    # print(f'***********\n\nepisode: {episode} | step: {step}\n\n**********')
    # print(f'{q_table}\n**********\n\n')

np.savetxt(fname='data/frozen_lake_q_table.csv', X=q_table, delimiter=',')
