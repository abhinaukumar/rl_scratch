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
import itertools

from mpl_toolkits.mplot3d import axes3d, Axes3D
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
        
def test(eps_env):
    #eps_env,scale = args
    env = ArgmaxEnv(eps_env)
    agent = Agent(env)
    corrects = []
    rewards = []
    lengths = []
    
    for scale in range(1,11):
        print "scale {} in eps {}".format(scale,eps_env)
        _corrects = []
        _lengths = []
        _rewards = []
        for i in range(5,10):   
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
                
            _corrects.append(correct/50.0)
            _lengths.append(length/5000.0)
            _rewards.append(avg_reward/5000.0)
            env.reset()
        corrects.append(_corrects)
        lengths.append(_lengths)
        rewards.append(_rewards)
        
    return lengths,corrects,rewards

if __name__ == '__main__':
    
    p = Pool(10)
    #scales = [7,6,10,5,8,1,1,1,1,1]
    ret = p.map(test,np.arange(10)/10.0)
    ret = np.array(ret)
    lengths = ret[:,0,:,:]
    corrects = ret[:,1,:,:]
    rewards = ret[:,2,:,:]
#    lengths = [r[0] for r in ret]
#    corrects = [r[1] for r in ret]
#    rewards = [r[2] for r in ret]

#    fig = plt.figure()
#    ax = fig.add_subplot(1,1,1, projection='3d')
#    ax.set_xlabel('Scale')
#    ax.set_ylabel('Threshold')
#    X,Y = np.meshgrid(np.arange(1,11),np.arange(5,10)/10.0)
#    
#    ax.set_title('Accuracy vs threshold')
#    for i in range(len(corrects)):
##        plt.plot(np.arange(10)/10.0,corrects[i],label="eps = %.2f, scale = %d" % (i/10.0, scales[i]))
#        ax.plot_surface(X,Y,corrects[i,:,:].T,rstride=1, cstride=1, linewidth=0, antialiased=False,label = 'eps = %.2f' % (i/10.0))
#    ax.legend()
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(1,1,1, projection='3d')
#    ax.set_xlabel('Scale')
#    ax.set_ylabel('Threshold')
#    
#    ax.set_title('Delay vs threshold')
#    for i in range(len(lengths)):
##        plt.plot(np.arange(10)/10.0,lengths[i],label="eps = %.2f, scale = %d" % (i/10.0, scales[i]))
#        ax.plot_surface(X,Y,lengths[i,:,:].T,rstride=1, cstride=1, linewidth=0, antialiased=False,label = 'eps = %.2f' % (i/10.0))
#    ax.legend()
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(1,1,1, projection='3d')
#    ax.set_xlabel('Scale')
#    ax.set_ylabel('Threshold')
#    
#    ax.set_title('Reward vs threshold')
#    for i in range(len(rewards)):
##        plt.plot(np.arange(10)/10.0,rewards[i],label="eps = %.2f, scale = %d" % (i/10.0, scales[i]))
#        ax.plot_surface(X,Y,rewards[i,:,:].T,rstride=1, cstride=1, linewidth=0, antialiased=False,label = 'eps = %.2f' % (i/10.0))
#    ax.legend()
    
    for i in range(10):
        scale,threshold = np.unravel_index(np.argmax(rewards[i,:,:], axis=None), rewards[i,:,:].shape)
        print "Optimal choice for environemt with epsilon {} is scale {} and threshold {} with average reward {}, accuracy {} and delay {}".format(i/10.0,1 + scale,(5 + threshold)/10.0,rewards[i,scale,threshold],corrects[i,scale,threshold],lengths[i,scale,threshold])

# Output
#    Optimal choice for environemt with epsilon 0.0 is scale 3 and threshold 0.5 with average reward 30.0, accuracy 100.0 and delay 1.0
#    Optimal choice for environemt with epsilon 0.1 is scale 9 and threshold 0.9 with average reward 25.188, accuracy 91.98 and delay 1.0
#    Optimal choice for environemt with epsilon 0.2 is scale 9 and threshold 0.5 with average reward 19.812, accuracy 83.02 and delay 1.0
#    Optimal choice for environemt with epsilon 0.3 is scale 5 and threshold 0.6 with average reward 14.436, accuracy 74.06 and delay 1.0
#    Optimal choice for environemt with epsilon 0.4 is scale 4 and threshold 0.5 with average reward 9.132, accuracy 65.22 and delay 1.0
#    Optimal choice for environemt with epsilon 0.5 is scale 2 and threshold 0.8 with average reward 6.56466417589, accuracy 95.04 and delay 4.78
#    Optimal choice for environemt with epsilon 0.6 is scale 1 and threshold 0.6 with average reward 3.80622162271, accuracy 96.54 and delay 7.7492
#    Optimal choice for environemt with epsilon 0.7 is scale 1 and threshold 0.8 with average reward 1.95030450486, accuracy 97.9 and delay 14.6432
#    Optimal choice for environemt with epsilon 0.8 is scale 1 and threshold 0.8 with average reward -1.20829582454, accuracy 90.2 and delay 21.9654
#    Optimal choice for environemt with epsilon 0.9 is scale 1 and threshold 0.7 with average reward -12.8527502214, accuracy 54.1 and delay 27.232

        