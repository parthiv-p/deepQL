#TODO: Change the np.zeroes logic to end episode to depend on 'done'
#TODO: decay for epsilon


import gym
import numpy as np
#for now I will use keras but I will switch it to TF once its working
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import random

MAX_SIZE = 100	#for history buffer
BATCH_SIZE = 32
MAX_STEPS = 199 #beyond this we consider solved
GAMMA = 0.99
ALPHA = 0.01
EPSILON = 0.8
env = gym.make('CartPole-v0')



def build_model():
	model = Sequential()
	model.add(Dense(24, activation = 'relu', input_dim = 4))
	model.add(Dense(24, activation='relu'))
	model.add(Dense(2, activation='linear'))
	model.compile(loss='mse', optimizer = Adam(lr = ALPHA))
	return model

def choose_action(model, state, action):
	return env.action_space.sample() if np.random.rand(1) <= EPSILON else np.argmax(model.predict(state))

def experience_replay(model):
	#sample batch from buffer 
	X_b, y_b = [], []
	
	minibatch = random.sample(replayBuffer, BATCH_SIZE) #unfortunately np.random.choice does not take 2D array

	for state, action, reward, next_state, done in minibatch:
		y = model.predict(state)  #this will be (1,2) shape
		y[0][action] = reward if done else reward + GAMMA * np.amax(model.predict(next_state)[0]) 	
		y_b.append(y[0])
		X_b.append(state[0])

	model.fit(np.array(X_b), np.array(y_b), batch_size = BATCH_SIZE)
	#logic for epsilon decay
	
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
				replayBuffer.append((state, action, reward, next_state, done))
				break
			replayBuffer.append((state, action, reward, next_state, done))
			state = next_state
	print ('Memory Prepopulated!')
		
def simulate(model, episodes):
	for episode in range(episodes):
		init_state = env.reset()
		init_state = np.reshape(init_state, [1, 4]) 
		state = init_state

		for step in range(MAX_STEPS):
			env.render()

			action = choose_action(model, state, EPSILON)
			observation, reward, done, info = env.step(action)
			next_state = np.reshape(observation, [1, 4])
			replayBuffer.append((state, action, reward, next_state, done))
			
			if done: 
				print ('Episode ended after {} time steps'.format(step))
				break

			state = next_state
		
		experience_replay(model)		
		#logic for explore rate decay

if __name__ == '__main__':
	model = build_model()
	replayBuffer = deque(maxlen = MAX_SIZE)
	populate_memory(model)
	simulate(model, episodes = 200)