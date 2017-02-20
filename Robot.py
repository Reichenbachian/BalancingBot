import pandas as pd
import numpy as np
import random
#import matplotlib.pyplot as plt
import serial

class Learner:
    def __init__(self):
        num_bins = 10
        num_states = 2
        num_states = num_bins ** num_states
        num_actions = 10
        self.ACTIONS = np.linspace(-1.5, 1.5, num_actions)
        self.qmatrix = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))
        self.epsilon = .6
        self.gamma = .9
        self.alpha = .2
        self.epsilonDecay = .99
        self.state = 0
        self.actionIndex = None

    def set_initial(self, state):
        self.state = state
        self.actionIndex = self.qmatrix[state].argsort()[-1]
        return self.ACTIONS

    def getMove(self, newState, reward):
        if random.random() < self.epsilon:
            return random.randint(0,num_actions)
        newActionIndex = self.qmatrix[newState].argsort()[-1]
        self.qmatrix[self.state, self.actionIndex] = (1 - self.alpha) * self.qmatrix[self.state, self.actionIndex] + \
                                                    self.alpha*(reward + self.gamma * self.qmatrix[newState, newActionIndex])
        self.actionIndex = newActionIndex
        self.state = newState
        self.epsilon *= self.epsilonDecay
        return self.ACTIONS[newActionIndex]

def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]

def getAngle():
    a=float('inf')
    while a == float('inf'):
        try:
            a = int(ser.readline().decode().strip())
        except:
            print("Failed reading...Trying again...")

def cart_pole_with_qlearning():
    learner = Learner()

    timeSurvived = []

    currentAngleBins = pd.cut([-50, 50], bins=10, retbins=True)[1][1:-1]
    previousAngleBins = pd.cut([-50, 50], bins=10, retbins=True)[1][1:-1]
    for episode in range(50000):
        previousAngle = getAngle()
        currentAngle = getAngle()
        state = build_state([to_bin(currentAngleBins, currentAngle),
                             to_bin(previousAngleBins, previousAngle)])

        action = learner.set_initial(state)
        for step in range(10000):
            previousAngle = getAngle()
            currentAngle = getAngle()
            newState = build_state([to_bin(currentAngleBins, currentAngle),
                             to_bin(previousAngleBins, previousAngle)])

            if done:
                reward = -200

            action = learner.getMove(newState, reward)

            #Do move

            if done:
                timeSurvived.append(step)
                break

if __name__ == "__main__":
    cart_pole_with_qlearning()