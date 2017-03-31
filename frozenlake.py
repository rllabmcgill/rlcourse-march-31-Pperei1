import gym
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

class Sarsa:

	def __init__(self,env,numberOfStates,numberOfActions):
		self.env = env
		self.numberOfStates = numberOfStates
		self.numberOfActions = numberOfActions
		self.valueFunction = np.zeros((self.numberOfStates,self.numberOfActions))

	def reset(self):
		self.valueFunction = np.zeros((self.numberOfStates,self.numberOfActions))

	def notValidMove(self,previousState,action):
		if action == 0:
			return previousState%4 == 0
		elif action == 1:
			return previousState > 11
		elif action == 2:
			return previousState%4 == 3
		else:
			return previousState < 4

	def nextState(self,previousState,action):
		"""
		0 -> left
		1 -> down
		2 -> right
		3 -> up
		"""
		if action == 0:
			return previousState - 1
		elif action == 1:
			return previousState + 4
		elif action == 2:
			return previousState + 1
		else:
			return previousState - 4

	def policyFunction(self,currentState,epsilon):
		bestValue = 0
		if currentState < 12:
			bestAction = 1
		else:
			bestAction = 3
		for i in range(0,4):
			if (not self.notValidMove(currentState,i)) and (bestValue < self.valueFunction[currentState][i]):
				bestAction = i;
				bestValue = self.valueFunction[currentState][i]
		if rnd.random() < epsilon:
			bestAction = self.env.action_space.sample()
		return bestAction

	def updateValue(self,previousState,previousAction,alpha,epsilon):
		observation, reward, done, info = env.step(previousAction)
		bestAction = self.policyFunction(observation,epsilon)
		self.valueFunction[previousState][previousAction] = self.valueFunction[previousState][previousAction] + alpha*((reward) + self.valueFunction[observation][bestAction] - self.valueFunction[previousState][previousAction])
		return done,observation,bestAction

	def training(self,episodes,learningRate,epsilon):
		for j in range(episodes):
			previousState = env.reset()
			previousAction = self.policyFunction(previousState,epsilon)
			while(True):
				done,previousState,previousAction = self.updateValue(previousState,previousAction,learningRate,epsilon)
				if done:
					break
		return self.valueFunction

	def testing(self,episodes):
		r = 0
		for j in range(episodes):
			previousState = env.reset()
			previousAction = self.policyFunction(previousState,0)
			while(True):
				previousState, reward, done, info = env.step(previousAction)
				r = r + reward
				if done:
					break
		return r/episodes

class QLearner:

	def __init__(self,env,numberOfStates,numberOfActions):
		self.env = env
		self.numberOfStates = numberOfStates
		self.numberOfActions = numberOfActions
		self.valueFunction = np.zeros((numberOfStates,numberOfActions))

	def reset(self):
		self.valueFunction = np.zeros((self.numberOfStates,self.numberOfActions))

	def notValidMove(self,previousState,action):
		if action == 0:
			return previousState%4 == 0
		elif action == 1:
			return previousState > 11
		elif action == 2:
			return previousState%4 == 3
		else:
			return previousState < 4

	def nextState(self,previousState,action):
		"""
		0 -> left
		1 -> down
		2 -> right
		3 -> up
		"""
		if action == 0:
			return previousState - 1
		elif action == 1:
			return previousState + 4
		elif action == 2:
			return previousState + 1
		else:
			return previousState - 4

	def policyFunction(self,currentState,epsilon):
		bestValue = 0
		if currentState < 12:
			bestAction = 1
		else:
			bestAction = 3
		for i in range(0,4):
			if (not self.notValidMove(currentState,i)) and (bestValue < self.valueFunction[currentState][i]):
				bestAction = i;
				bestValue = self.valueFunction[currentState][i]
		if rnd.random() < epsilon:
			bestAction = self.env.action_space.sample()
		return bestAction

	def updateValue(self,previousState,previousAction,alpha,epsilon):
		observation, reward, done, info = env.step(previousAction)
		bestAction = self.policyFunction(observation,0)
		self.valueFunction[previousState][previousAction] = self.valueFunction[previousState][previousAction] + alpha*((reward) + self.valueFunction[observation][bestAction] - self.valueFunction[previousState][previousAction])
		return done,observation

	def training(self,episodes,learningRate,epsilon):
		for j in range(episodes):
			previousState = env.reset()
			previousAction = self.policyFunction(previousState,epsilon)
			while(True):
				done,previousState = self.updateValue(previousState,previousAction,learningRate,epsilon)
				previousAction = self.policyFunction(previousState,epsilon)
				if done:
					break
		return self.valueFunction

	def testing(self,episodes):
		r = 0
		for j in range(episodes):
			previousState = env.reset()
			previousAction = self.policyFunction(previousState,0)
			while(True):
				previousState, reward, done, info = env.step(previousAction)
				r = r + reward
				if done:
					break
		return r/episodes

def randomPlayer(env,episodes):
	r = 0
	for j in range(episodes):
		env.reset()
		while(True):
			previousState, reward, done, info = env.step(env.action_space.sample())
			r = r + reward
			if done:
				break
	return r/episodes


env = gym.make('FrozenLake-v0')
S = Sarsa(env,16,4)
Q = QLearner(env,16,4)
alpha = [0.001,0.005,0.01,0.05,0.1]

sarsaReward = []
QReward = []
random = []


for a in alpha:
	S.reset()
	Q.reset()
	S.training(10000,a,0.05)
	Q.training(10000,a,0.05)
	sarsaReward.append(S.testing(1000))
	QReward.append(Q.testing(1000))
	random.append(randomPlayer(env,1000))



plt.plot(alpha,sarsaReward, label="sarsa")
plt.plot(alpha,QReward, label="Q")
plt.plot(alpha,random, label="Random")
plt.ylabel('AverageReward')
plt.xlabel('LearningRate')
 #Place a legend to the right of this smaller subplot.
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.show()

'''
S.reset()
Q.reset()
valueSarsa = S.training(1000,0.01,0.05)
valueQ = Q.training(1000,0.05,0.05)
for i in range(0,len(valueSarsa)):
	print valueSarsa[i]

for i in range(0,len(valueSarsa)):
	print valueQ[i]
'''