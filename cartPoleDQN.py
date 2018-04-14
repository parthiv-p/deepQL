import gym
import numpy as np
#for now I will use keras but I will switch it to TF once its working
import keras
from collections import deque

MAX_SIZE = 100	#for history buffer
BATCH_SIZE = 32
MAX_STEPS = 199 #beyond this we consider solved

env = gym.make('CartPole-v0')



replayBuffer = deque(maxlen = MAX_SIZE)



def build_model():
	model = Sequential()
	model.add(Dense(24, actitvation = 'relu', input_dim = 4))
	model.add(Dense(24, actitvation='relu'))
	model.add(Dense(2, actitvation='linear'))
	model.compile(loss='mse', optimizer = Adam(lr))
	return model

def choose_action(model, state, action):
	return env.action_space.sample() if np.random.rand(1) <= epsilon else np.argmax(model.predict(state))

def experience_replay():
	pass
	
def populate_memory(model):
	for episode in range(BATCH_SIZE):
		init_state = env.reset()
		init_state = np.reshape(init_state, [1, 4]) 
		state = init_state

		for step in range(MAX_STEPS):
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			next_state = np.reshape(observation, [1, 4])

			if done: 
				next_state = np.zeros(state.shape) #empty next state
				replayBuffer.append((state, action, reward, next_state))
				break
			replayBuffer.append((state, action, reward, next_state))
			state = next_state
	print ('Memory Prepopulated!')
		
def simulate(model, episodes):
	for episode in range(episodes):
		init_state = env.reset()
		init_state = np.reshape(init_state, [1, 4]) 
		state = init_state

		for step in range(MAX_STEPS):
			env.render()

			action = choose_action(model, state, epsilon)
			observation, reward, done, info = env.step(action)
			next_state = np.reshape(observation, [1, 4])

			if done: 
				next_state = np.zeros(state.shape) #empty next state
				replayBuffer.append((state, action, reward, next_state))
				print ('Episode ended after {} time steps'.format(step))
				break

		replayBuffer.append((state, action, reward, next_state))
		state = next_state
		
		experience_replay()		
		#logic for explore rate decay

if __name__ == '__main__':
	model = build_model()
	populate_memory(model)
	simulate(model, episodes)