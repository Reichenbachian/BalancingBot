import pandas as pd
import numpy as np
import random
import serial
import rospy
import roslib
from geometry_msgs.msg import Twist

# initializing various in order to control the robot
ser = serial.Serial('/dev/ttyACM0', 9600)
rospy.init_node('goalieBot', anonymous=True)
movement = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=5)
r = rospy.Rate(50);

class Learner:
    def __init__(self):
        num_bins = 10
        num_states = 2
        num_states = num_bins ** num_states
        num_actions = 100
        self.ACTIONS = np.linspace(-1.5, 1.5, num_actions)
        self.qmatrix = np.zeros((num_states, num_actions))
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
        if random.random() < self.epsilon: # allows for exploration (chooses a random action)
            return random.randint(0,num_actions)
        newActionIndex = self.qmatrix[newState].argsort()[-1]
        self.qmatrix[self.state, self.actionIndex] = (1 - self.alpha) * self.qmatrix[self.state, self.actionIndex] + \
                                                    self.alpha*(reward + self.gamma * self.qmatrix[newState, newActionIndex])
        self.actionIndex = newActionIndex
        self.state = newState
        self.epsilon *= self.epsilonDecay # decreases the need of randomness as the model becomes more trained
        return self.ACTIONS[newActionIndex]

def build_state(features):
    a = int("".join(map(lambda feature: str(int(feature)), features)))
    print("The action index is: ", a)
    return a

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]

def getAngle():
    a=float('inf')
    while a == float('inf'):
        try:
            a = int(ser.readline().decode().strip())
        except:
            print("Failed reading...Trying again...")

def makeMove(action):
    move_cmd = Twist()
    move_cmd.linear.x = action
    movement.publish(move_cmd)
    r.sleep()

def getReward():
    return 100/(abs(getAngle())**2+1)

def cart_pole_with_qlearning():
    learner = Learner()
    speed = 0
    currentAngleBins = np.array(list(range(-50,60,100)))
    previousAngleBins = np.array(list(range(-50,60,100)))
    previousAngle = getAngle()

    # we are not going to differentiate between episode and step because,
    # when we train the robot, we let it run without ever stopping it
    # at each episode (i.e. moment when robot fails)
    for step in range(10000):
        currentAngle = getAngle()
        state = build_state([to_bin(currentAngle, currentAngleBins),
                         to_bin(previousAngle, previousAngleBins)])
        reward = getReward()
        action = learner.getMove(state, reward)
        speed += action
        makeMove(speed)
        previousAngle = currentAngle

if __name__ == "__main__":
    cart_pole_with_qlearning()
