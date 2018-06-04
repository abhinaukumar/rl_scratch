# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:06:26 2018

@author: nownow
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 11:31:20 2018

@author: nownow
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# Define WindyGridworld environment

class CliffWalking():
    def __init__(self):
        #self.wind = np.array([[0,0],[0,0],[0,0],[-1,0],[-1,0],[-1,0],[-2,0],[-2,0],[-1,0],[0,0]])
        self.nrows = 4
        self.ncols = 12
        self.start_state = np.array([3,0])
        self.goal_state = np.array([3,11])
        self.nactions = 4
        self.actions = np.array([[-1,0],[1,0],[0,-1],[0,1]])#,[-1,1],[1,1],[1,-1],[-1,-1]])
        self.done = False
        self.cliff_left = 1
        self.cliff_right = 10
        
    def act(self,state,action):
        next_state = state + self.actions[action] #+ self.wind[state[1]] + np.array([np.random.choice([-1,0,1]),0])
        next_state = np.clip(next_state,0,[self.nrows-1,self.ncols-1])
        if next_state[0] == self.nrows - 1 and next_state[1] >= self.cliff_left and next_state[1] <= self.cliff_right:
            self.done = False
            return -100,self.start_state
        else:
            self.done = (next_state == self.goal_state).all()
            return int(not self.done)*-1,next_state
    
    def reset(self):
        self.done = False

# Define an epsilon greedy SARSA algorithm

def SARSA(gridworld,eps,alpha,gamma):
    Q = np.zeros((gridworld.nrows,gridworld.ncols,gridworld.nactions))
    step = 0
    steps = [0]
    r = []
    for i in range(500):
        state = gridworld.start_state
        e = np.random.uniform()
        r.append(0)
        action = np.argmax(Q[state[0],state[1],:]) if e >= eps else np.random.randint(0,gridworld.nactions)
        while not gridworld.done:
            e = np.random.uniform()
            reward,next_state = gridworld.act(state,action)
            r[-1] += reward
            next_action = np.argmax(Q[next_state[0],next_state[1],:]) if e >= eps else np.random.randint(0,gridworld.nactions)
            Q[state[0],state[1],action] += alpha*(reward + gamma*Q[next_state[0],next_state[1],next_action] - Q[state[0],state[1],action])
            state = next_state.copy()
            action = next_action
            step = step + 1
        steps.append(step)
        if not i%1000:
            print "Episode length: ",steps[-1] - steps[-2]
        gridworld.reset()
    return Q,steps,r

def q_learn(gridworld,eps,alpha,gamma):
    Q = np.zeros((gridworld.nrows,gridworld.ncols,gridworld.nactions))
    step = 0
    steps = [0]
    r = []
    for i in range(500):
        if i%10:
            eps*=0.95
        state = gridworld.start_state
        r.append(0)
        while not gridworld.done:
            e = np.random.uniform()
            action = np.argmax(Q[state[0],state[1],:]) if e >= eps else np.random.randint(0,gridworld.nactions)
            reward,next_state = gridworld.act(state,action)
            r[-1]+=reward
            Q[state[0],state[1],action] += alpha*(reward + gamma*np.argmax(Q[next_state[0],next_state[1],:]) - Q[state[0],state[1],action])
            state = next_state.copy()
            step = step + 1
        steps.append(step)
        if not i%100:
            print "Episode length: ",steps[-1] - steps[-2], "Exploration probability: ",eps
        gridworld.reset()
    return Q,steps,r
    
def expected_SARSA(gridworld,eps,alpha,gamma):
    Q = np.zeros((gridworld.nrows,gridworld.ncols,gridworld.nactions))
    p = np.ones((gridworld.nactions,))*eps/gridworld.nactions    
    step = 0
    steps = [0]
    r = []
    for i in range(500):
        state = gridworld.start_state
        r.append(0)
        while not gridworld.done:
            e = np.random.uniform()
            greedy_selection = np.argmax(Q[state[0],state[1],:])
            action = greedy_selection if e >= eps else np.random.randint(0,gridworld.nactions)
            reward,next_state = gridworld.act(state,action)
            r[-1]+=reward
            p_step = p.copy()
            p_step[greedy_selection] += 1 - eps
            Q[state[0],state[1],action] += alpha*(reward + gamma*np.dot(Q[next_state[0],next_state[1],:],p_step) - Q[state[0],state[1],action])
            state = next_state.copy()
            step = step + 1
        steps.append(step)
        if not i%1000:
            print "Episode ", i, "has length: ",steps[-1] - steps[-2]
        gridworld.reset()
    return Q,steps,r
    
gridworld = CliffWalking()
Q,steps,r1 = expected_SARSA(gridworld,0.1,0.5,1.0)

gridworld.reset()
state = gridworld.start_state
time = 0
while not gridworld.done:
    action = np.argmax(Q[state[0],state[1],:])
    state = gridworld.act(state,action)[-1]
    #print gridworld.actions[action]
    time+=1
print "Expected Sarsa: ",time

gridworld.reset()
Q,steps,r2 = SARSA(gridworld,0.1,0.5,1.0)

gridworld.reset()
state = gridworld.start_state
time = 0
while not gridworld.done:
    action = np.argmax(Q[state[0],state[1],:])
    state = gridworld.act(state,action)[-1]
    #print gridworld.actions[action]
    time+=1
print "Sarsa: ",time

plt.figure()
plt.plot(np.arange(10,len(r1)),r1[10:],label='Expected Sarsa')
plt.plot(np.arange(10,len(r2)),r2[10:],label='Sarsa')
plt.legend()