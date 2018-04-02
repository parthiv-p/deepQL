'''
Implementation of CartPole using Q learning 
Inspired from https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947
Its a great post I recommend you check it out
'''
import gym 
import numpy as np

BUCKETS = (1,1,6,3)	#constrain the continuous feature space to discrete values

env = gym.make('CartPole-v0')

#Constraint the observation space to reduce learning complexity
new_obs_space_low = [env.observation_space.low[0], -0.5,  env.observation_space.low[2], -np.radians(50)]
new_obs_space_high = [env.observation_space.high[0], 0.5,  env.observation_space.high[2], np.radians(50)]

#Create Q Table
Q_table = np.zeros(BUCKETS + (env.action_space.n,)) 

def map_state(state):
    new_state = []
    for i in range(len(state)):

        if state[i] <= new_obs_space_low[i]: mapped_index = 0
        elif state[i] >= new_obs_space_high[i]: mapped_index = BUCKETS[i] - 1
        else:
        	# We need to map X from the range (A,B) to (C,D)
            # this can be done by Y = (X-A)/(B-A) * (D-C) + C
            # we then round it to get the value as an int
            obs_space = (new_obs_space_low[i], new_obs_space_high[i])
            mapping_space = (0, BUCKETS[i] - 1)
            mapped_index = int(round(((state[i] - obs_space[0]) / (obs_space[1]-obs_space[0])) * (mapping_space[1]-mapping_space[0]) + mapping_space[0]))

        new_state.append(mapped_index)
    
    return tuple(new_state)

def choose_action(state, epsilon):
	return env.action_space.sample() if np.random.rand(1) <= epsilon else np.argmax(Q_table[state])

def update_Q_table(prev_state, next_state, action, reward, alpha, gamma):
	#standard formula to update q table
	Q_table[prev_state][action] += alpha * (reward + gamma * np.max(Q_table[next_state]) - Q_table[prev_state][action])

def simulate(N_EPISODES, MAX_STEPS, epsilon, alpha, gamma):
	for episode in range(N_EPISODES):

		init_state = map_state(env.reset())
		state = init_state
		for step in range(MAX_STEPS):
			env.render()
			action = choose_action(state, epsilon)
			observation, reward, done, info = env.step(action)
			next_state = map_state(observation)
			update_Q_table(state, next_state, action, reward, alpha, gamma)
			state = next_state
			if done:
				print("Episode finished after {} timesteps".format(step+1))
				break


if __name__ == '__main__':
	simulate(N_EPISODES = 100,
			MAX_STEPS = 200,
			epsilon = 0.9,
			alpha = 0.5,
			gamma = 0.9)