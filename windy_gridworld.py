# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 11:31:20 2018

@author: nownow
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# Define WindyGridworld environment

class WindyGridworld():
    def __init__(self):
        self.wind = np.array([[0,0],[0,0],[0,0],[-1,0],[-1,0],[-1,0],[-2,0],[-2,0],[-1,0],[0,0]])
        self.nrows = 7
        self.ncols = 10
        self.start_state = np.array([3,0])
        self.goal_state = np.array([3,7])
        self.nactions = 8
        self.actions = np.array([[-1,0],[1,0],[0,-1],[0,1],[-1,1],[1,1],[1,-1],[-1,-1]])
        self.done = False
        
    def act(self,state,action):
        next_state = state + self.actions[action] + self.wind[state[1]] + np.array([np.random.choice([-1,0,1]),0])
        next_state = np.clip(next_state,0,[self.nrows-1,self.ncols-1])
        self.done = (next_state == self.goal_state).all()
        return int(not self.done)*-1,next_state
    
    def reset(self):
        self.done = False

# Define an epsilon greedy SARSA algorithm

def SARSA(gridworld,eps,alpha,gamma):
    Q = np.zeros((gridworld.nrows,gridworld.ncols,gridworld.nactions))
    step = 0
    steps = [0]
    l = []
    for i in range(10000):
        state = gridworld.start_state
        e = np.random.uniform()
        action = np.argmax(Q[state[0],state[1],:]) if e >= eps else np.random.randint(0,gridworld.nactions)
        while not gridworld.done:
            reward,next_state = gridworld.act(state,action)
            next_action = np.argmax(Q[next_state[0],next_state[1],:]) if e >= eps else np.random.randint(0,gridworld.nactions)
            Q[state[0],state[1],action] += alpha*(reward + gamma*Q[next_state[0],next_state[1],next_action] - Q[state[0],state[1],action])
            state = next_state.copy()
            action = next_action
            step = step + 1
        steps.append(step)
        if i%10:
            print "Episode length: ",steps[-1] - steps[-2]
        gridworld.reset()
        l.append(steps[-1] - steps[-2])
    return Q,steps,l

gridworld = WindyGridworld()
Q,steps,l = SARSA(gridworld,0.2,0.25,1.0)

plt.figure()
plt.plot(steps,np.arange(len(steps)))
plt.figure()
plt.plot(l)

times = []
for i in range(1000):
    gridworld.reset()
    state = gridworld.start_state
    times.append(0)
    while not gridworld.done:
        action = np.argmax(Q[state[0],state[1],:])
        state = gridworld.act(state,action)[-1]
        times[-1]+=1
print "Average time taken:",np.mean(times)