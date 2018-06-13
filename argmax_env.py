#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 13:38:53 2018

@author: nownow
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
from multiprocessing import Pool

plt.ion()

import time

def softmax(x,axis=None):
    return np.exp(x)/np.expand_dims(np.sum(np.exp(x),axis=-1),axis=-1)

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
            reward = 20.0/self.prediction_time if guess == self.true_option else -20.0
            done = True
            self.true_option = np.random.randint(0,self.n_options)
            self.iters+=1   
            self.update_dist()
            self.prediction_time = 0
            
        # Can only wait up to max_prediction_time number of steps
        elif self.prediction_time > self.max_prediction_time:
            reward = -20.0
            done = True
            self.true_option = np.random.randint(0,self.n_options)
            self.iters+=1   
            self.update_dist()
            self.prediction_time = 0
        else:
            reward = -1.0
        
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
        self.scales = np.arange(5,self.n_actions)/float(self.n_actions),np.array()
        #self.scales = np.linspace(0.75,2.0,10)
        np.random.shuffle(self.scales)
        self.decay = 0.95
        self.action_values = np.ones(self.n_actions,)
                
    def reset_channels(self):
        self.channels = np.zeros((self.n_options,))


def train(args):
    eps_env,alpha = args
    start_time = time.time()
    env = ArgmaxEnv(eps_env)
    agent = Agent(env)
    
    corrects = []
    lengths = []
    rewards = []
    eps = 1.0
    q = []
    q.append(agent.action_values.copy())
    
    avg_reward = 0
    reward = None
    evidence = env.reset()
    agent.reset_channels()
    correct = 0
    length=0
    while env.iters < 1000:
        length+=1
        choice = np.random.choice(np.arange(agent.n_actions)) # we start with all actions equally likely and randomly chosen
        agent.channels = agent.decay*agent.channels + agent.scales[choice]*evidence
        prob = softmax(agent.channels)
        guess = np.argmax(prob) if np.max(prob) > 0.5 else None
        reward,evidence,done = env.step(guess)
        avg_reward+=reward
        
        if done:
            agent.reset_channels()
            correct += int(reward != -20)
            
    avg_reward/=1000.0
    print "Initial accuracy is {} with average decision time {} and average reward {} in environment with eps {}".format(correct/10.0,length/1000.0,avg_reward,env.eps)
        
    for i in range(1000):
        eps*=0.99
        length = 0
        reward = None
        evidence = env.reset()
        agent.reset_channels()
        while env.iters < 1000:
            #e = np.random.uniform()
            #choice = np.argmax(agent.action_values) if e >= eps else np.random.randint(0,agent.n_actions)
            choice = np.random.choice(np.arange(agent.n_actions),p = softmax(agent.action_values))

            agent.channels = agent.decay*agent.channels + agent.scales[choice]*evidence # The scale chosen is a measure of the agent's confidence in the decision
            prob = softmax(agent.channels)
            guess = np.argmax(prob) if np.max(prob) > 0.5 else None
            reward,evidence,done = env.step(guess)
            if done:
                agent.reset_channels()
                agent.action_values[choice] += alpha*(reward - agent.action_values[choice]) # Q-learning update on termination
            else:
                agent.action_values[choice] += alpha*(reward + np.max(agent.action_values) - agent.action_values[choice]) # Q-learning update before termination
                
        if not i%50:
            
            # Report the performance of the greedy policy every 50 iterations
            
            avg_reward = 0
            reward = None
            evidence = env.reset()
            agent.reset_channels()
            correct = 0
            length = 0
            choice = np.argmax(agent.action_values)
            while env.iters < 1000:
                length+=1
                agent.channels = agent.decay*agent.channels + agent.scales[choice]*evidence
                prob = softmax(agent.channels)
                guess = np.argmax(prob) if np.max(prob) > 0.5 else None
                reward,evidence,done = env.step(guess)
                avg_reward+=reward
                
                if done:
                    agent.reset_channels()
                    correct += int(reward != -20)
            avg_reward/=1000.0   
            print "Iteration {} in environment with eps {}".format(i,env.eps)
            corrects.append(correct/10.0)
            lengths.append(length/1000.0)
            rewards.append(avg_reward)
            q.append(agent.action_values.copy())

    avg_reward = 0
    reward = None
    evidence = env.reset()
    agent.reset_channels()
    correct = 0
    length=0
    choice = np.argmax(agent.action_values)
    while env.iters < 1000:
        length+=1
        agent.channels = agent.decay*agent.channels + agent.scales[choice]*evidence
        prob = softmax(agent.channels)
        guess = np.argmax(prob) if np.max(prob) > 0.5 else None
        reward,evidence,done = env.step(guess)
        avg_reward+=reward
        
        if done:
            agent.reset_channels()
            correct += int(reward != -20)
    avg_reward/=1000.0
    corrects.append(correct/10.0)
    lengths.append(length/1000.0)
    rewards.append(avg_reward)
    env.reset()
    print "eps = {} done in time {}. Final accuracy is {} %".format(env.eps,time.time() - start_time,correct/10.0)
    
    return lengths,corrects,rewards,agent,q
   
# Trying out all scales to find true optimal policy
       
def test(env_eps):
    start_time = time.time()
    env = ArgmaxEnv(env_eps)
    agent = Agent(env)
    corrects = []
    rewards = []
    lengths = []
    
    for i in range(10):   
        avg_reward = 0
        reward = None
        evidence = env.reset()       
        correct = 0
        length = 0
        while env.iters < 3000:
            length+=1
            agent.channels = agent.decay*agent.channels + i*evidence/10.0
            prob = softmax(agent.channels)
            guess = np.argmax(prob) if np.max(prob) > 0.5 else None
            reward,evidence = env.step(guess)
            avg_reward+=reward
            if guess != None:
                agent.reset_channels()
                correct += int(reward > 0)
        avg_reward/=3000.0
        corrects.append(correct/30.0)
        lengths.append(length/3000.0)
        rewards.append(avg_reward)
        env.reset()
    print "eps = {} done in time {}".format(env.eps,time.time() - start_time)
    return lengths,corrects,rewards

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='ArgMax Environment')
    parser.add_argument('--eps', type=float, default=None, metavar='E',
                        help='Spead of the distribution')
    args = parser.parse_args()
    
    if args.eps != None:
        ret = train((args.eps,0.01))
        plt.figure()
        plt.plot(ret[0],label="eps = %.2f" % (args.eps))
        plt.figure()
        plt.plot(ret[1],label="eps = %.2f" % (args.eps))
        plt.figure()
        plt.plot(ret[2],label="eps = %.2f" % (args.eps))
    else:
        p = Pool(10)
        ret = p.map(train,zip(np.arange(10)/10.0,np.ones((10,))*0.01))
        
        lengths = [r[0] for r in ret]
        corrects = [r[1] for r in ret]
        rewards = [r[2] for r in ret]
        scales = [r[3].scales[np.argmax(r[3].action_values)] for r in ret]
    
        plt.figure()
        plt.plot(np.arange(10)/1.0,scales)
        
        plt.figure()
        for i in range(len(corrects)):
            plt.plot(corrects[i],label="eps = %.2f" % (i/10.0))
        plt.legend()
        
        plt.figure()
        for i in range(len(lengths)):
            plt.plot(lengths[i],label="eps = %.2f" % (i/10.0))
        plt.legend()
        
        plt.figure()
        for i in range(len(rewards)):
            plt.plot(rewards[i],label="eps = %.2f" % (i/10.0))
        plt.legend()
    
#if __name__ == '__main__':
#    p = Pool(10)
#    ret = p.map(test,np.arange(10)/10.0)
#    
#    lengths = [r[0] for r in ret]
#    corrects = [r[1] for r in ret]
#    rewards = [r[2] for r in ret]
#
#    plt.figure()
#    for i in range(len(corrects)):
#        plt.plot(np.arange(10)/10.0,corrects[i],label="eps = %.2f" % (i/10.0))
#    plt.legend()
#    
#    plt.figure()
#    for i in range(len(lengths)):
#        plt.plot(np.arange(10)/10.0,lengths[i],label="eps = %.2f" % (i/10.0))
#    plt.legend()
#    
#    plt.figure()
#    for i in range(len(corrects)):
#        plt.plot(np.arange(10)/10.0,rewards[i],label="eps = %.2f" % (i/10.0))
#    plt.legend()
