import gym
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

class Learner:
	def __init__(self):
		self.state = np.linspace(-10, 10, 100)
		self.moves = np.array([0, 1])
		self.qmatrix = np.array([[random.random() for x in range(len(self.state))] for y in range(len(self.moves))])
		self.epsilon = .6
		self.gamma = .9
		self.alpha = .2
		self.epsilonDecay = .99
		self.actionIndex = None

	def set_initial(self, state):
		self.state = state
		# print("ACTION:", newAction)
		return self.moves[int(random.random()*len(self.moves))]

	def getMove(self, newState, reward):
		if random.random() < self.epsilon or self.actionIndex == None:
			# print("Doing random action...")
			return self.moves[int(random.random()*len(self.moves))]
		newAction = self.qmatrix[newState].argsort()[-1]
		stateString = ''.join(self.state)
		self.qmatrix[stateString,actionIndex] = self.qmatrix[stateString, actionIndex] + \
													self.alpha*(reward + gamma * self.qmatrix[newState, newAction])
		self.actionIndex = actionIndex
		self.state = newState
		action *= self.epsilonDecay
		return newAction


def cart_pole_with_qlearning():
    env = gym.make('CartPole-v0')
    experiment_filename = './cartpole-experiment-1'
    env = gym.wrappers.Monitor(env, experiment_filename, force=True)
    learner = Learner()

    timeSurvived = []

    for episode in range(50000):
        try:
            observation = env.reset()
        except:
            pass
        state = list(observation)

        action = learner.set_initial(state)
        for step in range(1000):
            observation, reward, done, info = env.step(action)

            state = list(observation)

            if done:
                reward = -200

            action = learner.getMove(state, reward)

            if done:
                timeSurvived.append(step)
                break
        # plt.plot(timeSurvived)
        # plt.pause(0.05)
        # print(timeSurvived)
        # if last_time_steps.mean() > goal_average_steps:
        #     print("Goal reached!")
        #     print("Episodes before solve: ", episode + 1)
        #     print(u"Best 100-episode performance {} {} {}".format(last_time_steps.max(),
        #                                                           unichr(177),  # plus minus sign
        #                                                           last_time_steps.std()))
        #     break
    env.monitor.close()

if __name__ == "__main__":
    random.seed(0)
    # plt.ion()
    cart_pole_with_qlearning()