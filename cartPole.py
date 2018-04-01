'''
Implementation of CartPole using Q learning 
Inspired from https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947
Its a great post I recommend you check it out
'''
import gym 
import numpy as np

BUCKETS = (1,1,6,3)
N_EPISODES = 12

env = gym.make('CartPole-v0')

#Constraint the observation space to reduce learning complexity
new_obs_space_low = [env.observation_space.low[0], -0.5,  env.observation_space.low[2], -np.radians(50)]
new_obs_space_high = [env.observation_space.high[0], 0.5,  env.observation_space.high[2], np.radians(50)]

def map_state(state):
    new_state = []
    for i in range(len(state)):

        if state[i] <= new_obs_space_low[i]: mapped_index = 0
        elif state[i] >= new_obs_space_high[i]: mapped_index = NUM_BUCKETS[i] - 1
        else:
        	# We need to map X from the range (A,B) to (C,D)
            # this can be done by Y = (X-A)/(B-A) * (D-C) + C
            # we then round it to get the value as an int
            obs_space = (new_obs_space_low[i], new_obs_space_high[i])
            mapping_space = (0, BUCKETS[i] - 1)
            mapped_index = int(round(((state[i] - obs_space[0]) / (obs_space[1]-obs_space[0])) * (mapping_space[1]-mapping_space[0]) + mapping_space[0]))

        new_state.append(mapped_index)
    
    return new_state   #might need to typecast to tuple

# for episode in range(N_EPISODES):

init_state = map_state(env.reset())
