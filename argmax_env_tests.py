#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 15:22:19 2018

@author: nownow
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
from multiprocessing import Pool
import itertools

from mpl_toolkits.mplot3d import axes3d, Axes3D
plt.ion()

import time

font = {'size': 13}

matplotlib.rc('font', **font)

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
        self.max_prediction_time = 30
    
    # Updates the source distribution 
    def update_dist(self):
        #self.dist = np.ones((self.n_options,))*self.eps/self.n_options
        self.dist = np.zeros((self.n_options,))
        self.dist[self.true_option] = 1 - self.eps
        self.dist[:self.true_option] = self.eps/(self.n_options - 1)
        self.dist[self.true_option+1:] = self.eps/(self.n_options - 1)
        
    # Step through the environment.
    def step(self,guess):
        self.prediction_time+=1
        done = False
        if guess != None:
            reward = 30.0 - (self.prediction_time - 1) if guess == self.true_option else -30.0
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
    
    for scale in range(1,6):
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
            while env.iters < 10000:
                length+=1
                agent.channels = agent.decay*agent.channels + scale*evidence
                prob = softmax(agent.channels)
                guess = np.argmax(prob) if np.max(prob) > i/10.0 else None
                reward,evidence,done = env.step(guess)
                avg_reward+=reward
                
                if done:
                    agent.reset_channels()
                    correct += int(reward != -30)
                
            _corrects.append(correct/100.0)
            _lengths.append(length/10000.0)
            _rewards.append(avg_reward/10000.0)
            env.reset()
        corrects.append(_corrects)
        lengths.append(_lengths)
        rewards.append(_rewards)
        
    return lengths,corrects,rewards

if __name__ == '__main__':
    
    p = Pool(5)
    #scales = [7,6,10,5,8,1,1,1,1,1]
    ret = p.map(test,np.arange(0,10,2)/10.0)
    ret = np.array(ret)
    lengths = ret[:,0,:,:]
    corrects = ret[:,1,:,:]
    rewards = ret[:,2,:,:]
    
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1, projection='3d')
#    ax1.axis('square')
    ax1.set_xticks(np.arange(1.0,6.0,1.0))
    ax1.set_yticks(np.arange(0.5,1.0,0.1))
    ax1.set_xlabel(r'Scale ($\alpha$)')
    ax1.set_ylabel('Threshold ($T$)')
    ax1.set_zlabel('Accuracy in %')
    X,Y = np.meshgrid(np.arange(1,6),np.arange(5,10)/10.0)
    
    ax1.set_title('Accuracy vs scale and threshold')
    for i in range(len(corrects)):
    #        plt.plot(np.arange(10)/10.0,corrects[i],label="eps = %.2f, scale = %d" % (i/10.0, scales[i]))
        ax1.plot_surface(X,Y,corrects[i,:,:].T,rstride=1, cstride=1, linewidth=0, antialiased=True,label = '$\epsilon$ = %.2f' % (i/5.0),color = colours[i])
    #ax.legend()
    
    fig = plt.figure()
    ax2 = fig.add_subplot(1,1,1, projection='3d')
#    ax2.axis('square')
    ax2.set_xticks(np.arange(1.0,6.0,1.0))
    ax2.set_yticks(np.arange(0.5,1.0,0.1))
    ax2.set_xlabel(r'Scale ($\alpha$)')
    ax2.set_ylabel('Threshold ($T$)')
    ax2.set_zlabel('Decision time')
    ax2.set_title('Decision time vs scale and threshold')
    for i in range(len(lengths)):
    #        plt.plot(np.arange(10)/10.0,lengths[i],label="eps = %.2f, scale = %d" % (i/10.0, scales[i]))
        ax2.plot_surface(X,Y,lengths[i,:,:].T,rstride=1, cstride=1, linewidth=0, antialiased=True,label = '$\epsilon$ = %.2f' % (i/5.0),color = colours[i])
    #ax.legend()
    
    fig = plt.figure()
    ax3 = fig.add_subplot(1,1,1, projection='3d')
    ax3.set_xticks(np.arange(1.0,6.0,1.0))
    ax3.set_yticks(np.arange(0.5,1.0,0.1))
#    ax3.axis('square')
    ax3.set_xlabel(r'Scale ($\alpha$)')
    ax3.set_ylabel('Threshold ($T$)')
    ax3.set_zlabel('Reward')
    ax3.set_title('Reward vs scale and threshold')
    for i in range(len(rewards)):
    #        plt.plot(np.arange(10)/10.0,rewards[i],label="eps = %.2f, scale = %d" % (i/10.0, scales[i]))
        ax3.plot_surface(X,Y,rewards[i,:,:].T,rstride=1, cstride=1, linewidth=0, antialiased=True,label = '$\epsilon$ = %.2f' % (i/5.0),color = colours[i])
    #ax.legend()
    
    for i in range(5):
        scale,threshold = np.unravel_index(np.argmax(rewards[i,:,:], axis=None), rewards[i,:,:].shape)
        print "Optimal choice for environment with epsilon {} is scale {} and threshold {} with average reward {}, accuracy {} and delay {}".format(i/5.0,1 + scale,(5 + threshold)/10.0,rewards[i,scale,threshold],corrects[i,scale,threshold],lengths[i,scale,threshold])

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


# After modifying environment
#
#    Optimal choice for environment with epsilon 0.0 is scale 1 and threshold 0.0 with average reward 30.0, accuracy 100.0 and delay 1.0
#    Optimal choice for environment with epsilon 0.1 is scale 4 and threshold 0.9 with average reward 28.6514, accuracy 99.73 and delay 2.192
#    Optimal choice for environment with epsilon 0.2 is scale 2 and threshold 0.8 with average reward 27.9099, accuracy 99.54 and delay 2.8208
#    Optimal choice for environment with epsilon 0.3 is scale 2 and threshold 0.8 with average reward 26.9633, accuracy 98.76 and delay 3.3173
#    Optimal choice for environment with epsilon 0.4 is scale 2 and threshold 0.9 with average reward 25.7409, accuracy 99.39 and delay 4.9208
#    Optimal choice for environment with epsilon 0.5 is scale 2 and threshold 0.9 with average reward 24.171, accuracy 98.33 and delay 5.9132
#    Optimal choice for environment with epsilon 0.6 is scale 2 and threshold 0.9 with average reward 21.2842, accuracy 95.75 and delay 7.4413
#    Optimal choice for environment with epsilon 0.7 is scale 2 and threshold 0.9 with average reward 16.0094, accuracy 89.72 and delay 9.85
#    Optimal choice for environment with epsilon 0.8 is scale 1 and threshold 0.6 with average reward 6.5203, accuracy 76.44 and delay 14.1006
#    Optimal choice for environment with epsilon 0.9 is scale 1 and threshold 0.5 with average reward -9.8193, accuracy 42.06 and delay 14.1068


# truth vs lie

#    Optimal choice for environment with epsilon 0.0 is scale 3 and threshold 0.5 with average reward 30.0, accuracy 100.0 and delay 1.0
#    Optimal choice for environment with epsilon 0.2 is scale 2 and threshold 0.8 with average reward 27.7538, accuracy 99.47 and delay 2.9374
#    Optimal choice for environment with epsilon 0.4 is scale 2 and threshold 0.9 with average reward 25.1314, accuracy 99.01 and delay 5.32
#    Optimal choice for environment with epsilon 0.6 is scale 1 and threshold 0.6 with average reward 18.406, accuracy 93.46 and delay 9.3273
#    Optimal choice for environment with epsilon 0.8 is scale 1 and threshold 0.5 with average reward -7.9188, accuracy 45.9 and delay 13.8281
