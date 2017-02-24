import pandas as pd
import numpy as np
import random
import serial
import rospy
import roslib
from geometry_msgs.msg import Twist
import pickle
from threading import Thread
import time

# initializing various in order to control the robot
ser = serial.Serial('/dev/ttyACM0', 9600)
rospy.init_node('goalieBot', anonymous=True)
movement = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=None, tcp_nodelay=True, latch=True)
timePerStep = 4
r = rospy.Rate(timePerStep)
moveCmd = Twist()
speed = 0

class Learner:
    num_bins = 10
    num_actions = 2
    def __init__(self):
        num_states = 2
        self.qmatrix = np.zeros((Learner.num_bins+1, Learner.num_bins+1, Learner.num_actions+1))
        self.epsilon = .7
        self.gamma = .9
        self.alpha = .2
        self.epsilonDecay = .99
        self.state = [0, 0]
        self.actionIndex = 0

    def set_initial(self, state):
        self.state = state
        self.actionIndex = self.qmatrix[state[0]][state[1]].argsort()[-1]
        return self.ACTIONS

    def getMove(self, newState, reward):
        if random.random() < self.epsilon: # allows for exploration (chooses a random action)
            return random.randint(0,Learner.num_actions)
        newActionIndex = self.qmatrix[newState[0]][newState[1]].argsort()[-1]
        #alpha = 1
        self.qmatrix[int(self.state[0])][int(self.state[1])][int(self.actionIndex)] = float(self.qmatrix[self.state[0]][self.state[1]][self.actionIndex]) + \
                                    reward + self.gamma * float(self.qmatrix[int(newState[0])][int(newState[1])][int(newActionIndex)])
        self.actionIndex = newActionIndex
        self.state = newState
        self.epsilon *= self.epsilonDecay # decreases the need of randomness as the model becomes more trained
        return newActionIndex
        self.epsilon *= self.epsilonDecay # decreases the need of randomness as the model becomes more trained
        return newActionIndex

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]

def getAngle():
    a=float('inf')
    while a == float('inf'):
        try:
            a = int(ser.readline().decode().strip())
        except:
            print("Failed reading...Trying again...")
    return a

def makeMove(action):
    global moveCmd
    current = 0
    current += action/100
    moveCmd = Twist()
    moveCmd.linear.x = action
    movement.publish(moveCmd)

def getReward():
    # getAngle() + 
    print(100/(abs(speed*30)**2+1))
    return 100/(abs(speed*30)**2+1)

def cart_pole_with_qlearning():
    global speed
    learner = Learner()
    angleBins = np.array(list(np.linspace(-60,70,Learner.num_bins)))
    actionBins = np.array(list(np.linspace(-1,1,Learner.num_actions+1)))
    previousAngle = getAngle()

    try:
        print("Trying to load matrix...", end='')
        learner.qmatrix = pickle.load(open("qMatrixAccel.pkl", 'rb'))
        print("Success!")
    except:
        print("Failed.")
    # we are not going to differentiate between episode and step because,
    # when we train the robot, we let it run without ever stopping it
    # at each episode (i.e. moment when robot fails)
    for step in range(10000):
        currentAngle = getAngle()
        state = [to_bin(currentAngle, angleBins), to_bin(previousAngle, angleBins)]
        reward = getReward()
        action = actionBins[learner.getMove(state, reward)]
        reward = getReward()
        action = actionBins[learner.getMove(state, reward)]
        thread = Thread(target = makeMove, args = (speed, ))
        thread.start()
        speed += .1 if action == 1 else -.1
        makeMove(speed)
        previousAngle = currentAngle
        time.sleep(1/timePerStep)
        if step%5 == 0:
            print("Saving the q-matrix")
            pickle.dump(learner.qmatrix, open("qMatrixAccel.pkl", 'wb'))

if __name__ == "__main__":
    cart_pole_with_qlearning()
