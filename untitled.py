import gym
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

class Learner:
    def __init__(self):
        num_bins = 10
        num_states = num_bins ** 4
        num_actions = 2
        self.qmatrix = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))
        self.epsilon = .6
        self.gamma = .9
        self.alpha = .2
        self.epsilonDecay = .99
        self.state = 0

    def set_initial(self, state):
        self.state = state
        self.action = self.qmatrix[state].argsort()[-1]
        return self.action

    def getMove(self, newState, reward):
        if random.random() < self.epsilon:
            return random.randint(0,1)
        newAction = self.qmatrix[newState].argsort()[-1]
        self.qmatrix[self.state, self.action] = (1 - self.alpha) * self.qmatrix[self.state, self.action] + \
                                                    self.alpha*(reward + self.gamma * self.qmatrix[newState, newAction])
        self.action = newAction
        self.state = newState
        self.epsilon *= self.epsilonDecay
        return newAction

def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]

def cart_pole_with_qlearning():
    env = gym.make('CartPole-v0')
    experiment_filename = './cartpole-experiment-1'
    env = gym.wrappers.Monitor(env, experiment_filename, force=True)
    learner = Learner()

    timeSurvived = []

    cart_position_bins = pd.cut([-2.4, 2.4], bins=10, retbins=True)[1][1:-1]
    pole_angle_bins = pd.cut([-2, 2], bins=10, retbins=True)[1][1:-1]
    cart_velocity_bins = pd.cut([-1, 1], bins=10, retbins=True)[1][1:-1]
    angle_rate_bins = pd.cut([-3.5, 3.5], bins=10, retbins=True)[1][1:-1]

    for episode in range(50000):
        try:
            observation = env.reset()
        except:
            pass
        cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
        state = build_state([to_bin(cart_position, cart_position_bins),
                             to_bin(pole_angle, pole_angle_bins),
                             to_bin(cart_velocity, cart_velocity_bins),
                             to_bin(angle_rate_of_change, angle_rate_bins)])

        action = learner.set_initial(state)
        for step in range(10000):
            observation, reward, done, info = env.step(action)
            cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
            newState = build_state([to_bin(cart_position, cart_position_bins),
                                       to_bin(pole_angle, pole_angle_bins),
                                       to_bin(cart_velocity, cart_velocity_bins),
                                       to_bin(angle_rate_of_change, angle_rate_bins)])

            if done:
                reward = -200

            action = learner.getMove(newState, reward)

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