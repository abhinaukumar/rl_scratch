#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 15:22:19 2018

@author: nownow
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from multiprocessing import Pool

plt.ion()

import time

def softmax(x,axis=None):
    return np.exp(x)/np.expand_dims(np.sum(np.exp(x),axis=axis),axis=-1)

class ArgmaxEnv():
    def __init__(self,eps):
        self.eps = eps
        self.n_options = 10
        self.true_option = np.random.randint(0,self.n_options)
        self.prediction_time = 0
        self.iters = 0
        self.dist = np.zeros((self.n_options,))
        self.update_dist()
        self.cur_state = 0
        self.max_prediction_time = 50
    
    # Updates the source distribution 
    def update_dist(self):
        self.dist = np.ones((self.n_options,))*self.eps/self.n_options
        self.dist[self.true_option] += 1 - self.eps
        
    # Step through the environment.
    def step(self,guess):
        self.prediction_time+=1
        done = False
        if guess != None:
            reward = 30.0/self.prediction_time if guess == self.true_option else -30.0
            done = True
            self.true_option = np.random.randint(0,self.n_options)
            self.iters+=1   
            self.update_dist()
            self.prediction_time = 0
            
        # Can only wait up to max_prediction_time number of steps
        elif self.prediction_time > self.max_prediction_time:
            reward = -30.0
            done = True
            self.true_option = np.random.randint(0,self.n_options)
            self.iters+=1   
            self.update_dist()
            self.prediction_time = 0
        else:
            reward = 0
        
        evidence = np.random.multinomial(1,self.dist)
        return (reward,evidence,done)
    
    def reset(self):
        self.true_option = np.random.randint(0,self.n_options)
        self.prediction_time = 0
        self.iters = 0
        self.update_dist()
        return np.random.multinomial(1,self.dist)
    
class Agent():
    def __init__(self,env):
        self.n_options = env.n_options
        self.n_actions = 10
        self.channels = np.zeros((self.n_options,))
        #self.scales = np.arange(self.n_actions)/float(self.n_actions)
        self.scales = np.arange(1.0,11.0)
        np.random.shuffle(self.scales)
        self.decay = 1.0
        #self.action_values = np.ones(self.n_actions,)
        self.action_values = np.random.normal(1.0,0.25,(self.n_actions,))
                
    def reset_channels(self):
        self.channels = np.zeros((self.n_options,))
        
def test(args):
    eps_env,scale = args
    env = ArgmaxEnv(eps_env)
    agent = Agent(env)
    corrects = []
    rewards = []
    lengths = []
    
    for i in range(10):   
        avg_reward = 0
        reward = None
        evidence = env.reset()
        agent.reset_channels()
        correct = 0
        length = 0
        while env.iters < 5000:
            length+=1
            agent.channels = agent.decay*agent.channels + scale*evidence
            prob = softmax(agent.channels)
            guess = np.argmax(prob) if np.max(prob) > i/10.0 else None
            reward,evidence,done = env.step(guess)
            avg_reward+=reward
            
            if done:
                agent.reset_channels()
                correct += int(reward != -30)
            
        corrects.append(correct/50.0)
        lengths.append(length/5000.0)
        rewards.append(avg_reward/5000.0)
        env.reset()
    return lengths,corrects,rewards

if __name__ == '__main__':
    
    p = Pool(10)
    ret = p.map(test,zip(np.arange(10)/10.0,[7,6,10,5,8,1,1,1,1,1]))
    
    lengths = [r[0] for r in ret]
    corrects = [r[1] for r in ret]
    rewards = [r[2] for r in ret]

    plt.figure()
    plt.title('Accuracy vs threshold')
    for i in range(len(corrects)):
        plt.plot(np.arange(10)/10.0,corrects[i],label="eps = %.2f" % (i/10.0))
    plt.legend()
    
    plt.figure()
    plt.title('Delay vs threshold')
    for i in range(len(lengths)):
        plt.plot(np.arange(10)/10.0,lengths[i],label="eps = %.2f" % (i/10.0))
    plt.legend()
    
    plt.figure()
    plt.title('Reward vs threshold')
    for i in range(len(rewards)):
        plt.plot(np.arange(10)/10.0,rewards[i],label="eps = %.2f" % (i/10.0))
    plt.legend()