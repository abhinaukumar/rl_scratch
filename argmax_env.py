#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 13:38:53 2018

@author: nownow
"""
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import time

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

class ArgmaxEnv():
    def __init__(self,eps):
        self.eps = eps
        self.n_options = 5
        self.true_option = np.random.randint(0,self.n_options)
        self.prediction_time = 0
        self.iters = 0
        self.dist = np.zeros((self.n_options,))
        self.update_dist()
        self.cur_state = 0
    
    def update_dist(self):
        self.dist = np.ones((self.n_options,))*self.eps/self.n_options
        self.dist[self.true_option] += 1 - self.eps
        
    def step(self,guess):
        self.prediction_time+=1
        
        if guess == self.true_option:
            self.true_option = np.random.randint(0,self.n_options)
            reward = 10.0/(self.prediction_time*0.5)
            self.prediction_time = 0
            self.iters+=1   
            self.update_dist()
            return reward
        elif guess is not None:
            self.true_option = np.random.randint(0,self.n_options)
            reward = -10.0
            self.prediction_time = 0
            self.iters+=1
            self.update_dist()
            return reward
        else:
            evidence = np.random.multinomial(1,self.dist)
            return (-0.5,evidence)
    
    def reset(self):
        self.true_option = np.random.randint(0,self.n_options)
        self.prediction_time = 0
        self.iters = 0
        self.update_dist()
        
class Agent():
    def __init__(self,env):
        self.n_options = env.n_options
        self.n_actions = 10
        self.channels = np.zeros((self.n_options,))
        self.thresholds = np.arange(self.n_actions)/float(self.n_actions)
        self.decay = 0.9
        self.action_values = np.ones(self.n_actions,)*10
                
    def reset_channels(self):
        self.channels = np.zeros((self.n_options,))


def train(agent,env):
    corrects = []
    lengths = []
    eps = 0.5
    counts = np.zeros(agent.n_actions)
    for i in range(1000):
        eps*=0.99
        correct = 0
        length = 0
        while env.iters < 500:
            length+=1
            e = np.random.uniform()
            choice = np.argmax(agent.action_values) if e >= eps else np.random.randint(0,agent.n_actions)
            prob = softmax(agent.channels)
            guess = np.argmax(prob) if np.max(prob) > agent.thresholds[choice] else None
            ret = env.step(guess)
            
            counts[choice]+=1
            if guess is None:
                agent.channels = agent.decay*agent.channels + ret[1]
                agent.action_values[choice] = agent.action_values[choice] + (ret[0]-agent.action_values[choice])/float(counts[choice])
            else:
                agent.action_values[choice] = agent.action_values[choice] + (ret-agent.action_values[choice])/float(counts[choice])
                agent.reset_channels()
                
                correct += int(ret > 0)
        if not i%50:
            print "{}/500 correct in thread with eps {} in iteration {}".format(correct,env.eps,i)
        corrects.append(correct/5.0)
        lengths.append(length/500.0)
        env.reset() 
        agent.reset_channels()
        
    env.reset()
        
    return lengths,corrects
          
if __name__ == '__main__':
    start_time = time.time()
    thresholds = []
    perfs = []
    lengths = []
    for i in range(10):
        print i
        env = ArgmaxEnv(i/10.0)
        agent = Agent(env)
        l,perf = train(agent,env)
        perfs.append(perf)
        lengths.append(l)
        q = agent.action_values
        thresholds.append(agent.thresholds[np.argmax(q)])
    plt.figure()
    plt.plot(np.arange(10)/10.0,thresholds)
    
    plt.figure()
    for i in range(len(perfs)):
        plt.plot(perfs[i],label="eps = %.2f" % (i/10.0))
    plt.legend()
    
    plt.figure()
    for i in range(len(lengths)):
        plt.plot(lengths[i],label="eps = %.2f" % (i/10.0))
    plt.legend()
    
    print "Time taken is {} seconds".format(time.time() - start_time)