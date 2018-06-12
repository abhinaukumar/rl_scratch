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
    return np.exp(x)/np.sum(np.exp(x),axis=axis)

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
    
    def update_dist(self):
        self.dist = np.ones((self.n_options,))*self.eps/self.n_options
        self.dist[self.true_option] += 1 - self.eps
        
    def step(self,guess):
        self.prediction_time+=1
        if guess != None:
            reward = 20.0/self.prediction_time if guess == self.true_option else -20.0
            self.true_option = np.random.randint(0,self.n_options)
            self.iters+=1   
            self.update_dist()
            self.prediction_time = 0
            
        elif self.prediction_time > self.max_prediction_time:
            reward = -20.0
            self.true_option = np.random.randint(0,self.n_options)
            self.iters+=1   
            self.update_dist()
            self.prediction_time = 0
        else:
            reward = - 0.9**((self.max_prediction_time - self.prediction_time)*0.5)
        
        evidence = np.random.multinomial(1,self.dist)
        return (reward,evidence)
    
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
        self.scales = np.arange(self.n_actions)/float(self.n_actions)
        np.random.shuffle(self.scales)
        self.decay = 0.9
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
    #counts = np.zeros(agent.n_actions)
    
    #R_av = 0.0
    #H = np.zeros((agent.n_actions,))
    #pi = softmax(H)
    
    avg_reward = 0
    reward = None
    evidence = env.reset()
    correct = 0
    length=0
    while env.iters < 1000:
        length+=1
        choice = np.random.choice(np.arange(agent.n_actions)) # we start with all actions equally likely
        agent.channels = agent.decay*agent.channels + agent.scales[choice]*evidence
        #print agent.channels
        prob = softmax(agent.channels)
        guess = np.argmax(prob) if np.max(prob) > 0.3 else None
        reward,evidence = env.step(guess)
        avg_reward+=reward
        #agent.action_values[choice] += alpha*(reward + np.max(agent.action_values) - agent.action_values[choice])
        
        if guess != None:   
            agent.reset_channels()
            correct += int(reward > 0)
        elif reward == -20:
            agent.reset_channels()
            
    avg_reward/=1000.0
    print "Initial accuracy is {} with average decision time {} and average reward {} in environment with eps {}".format(correct/10.0,length/1000.0,avg_reward,env.eps)
        
    for i in range(1000):
        eps*=0.99
        length = 0
        reward = None
        evidence = env.reset()
        
        while env.iters < 1000:
            e = np.random.uniform()
            choice = np.argmax(agent.action_values) if e >= eps else np.random.randint(0,agent.n_actions)
            #choice = np.random.choice(np.arange(agent.n_actions),p = pi)
            #counts[choice]+=1
            #prob = softmax(agent.channels,agent.scales[choice])
            #guess = np.argmax(prob) if np.max(prob) > 0.7 else None
            #ret = env.step(guess)
            agent.channels = agent.decay*agent.channels + agent.scales[choice]*evidence
            prob = softmax(agent.channels)
            guess = np.argmax(prob) if np.max(prob) > 0.5 else None
            reward,evidence = env.step(guess)
            #p = np.ones((agent.n_actions,))*eps/agent.n_actions
            #p[np.argmax(agent.action_values)] += 1 - eps
            #R_av = (reward+ R_av*step)/(step+1)
            
            agent.action_values[choice] += alpha*(reward + np.max(agent.action_values) - agent.action_values[choice])
            
            #H[choice] += alpha*(reward - R_av)*(1 - pi[choice])
            #H[:choice] -= alpha*(reward - R_av)*(pi[:choice])
            #H[choice+1:] -= alpha*(reward - R_av)*(pi[choice+1:])
        
            #pi = softmax(H)
            
            if guess != None:
                agent.reset_channels()
                
#            if guess is None:
#                agent.channels = agent.decay*agent.channels + agent.scales[choice]*ret[1]
#                prob = softmax(agent.channels,agent.scales[choice])
#                guess = np.argmax(prob) if np.max(prob) > 0.7 else None
#                agent.action_values[choice] = agent.action_values[choice] + (ret[0]-agent.action_values[choice])/float(counts[choice])
#                ret = env.step(guess)
#                if guess is None:
#                    
#            else:
#                agent.action_values[choice] = agent.action_values[choice] + (ret-agent.action_values[choice])/float(counts[choice])
#                agent.reset_channels()
        if not i%50:
            avg_reward = 0
            reward = None
            evidence = env.reset()
            correct = 0
            length=0
            while env.iters < 1000:
                length+=1
                choice = np.argmax(agent.action_values)
                agent.channels = agent.decay*agent.channels + agent.scales[choice]*evidence
                #print agent.channels
                prob = softmax(agent.channels)
                guess = np.argmax(prob) if np.max(prob) > 0.3 else None
                reward,evidence = env.step(guess)
                avg_reward+=reward
                #agent.action_values[choice] += alpha*(reward + np.max(agent.action_values) - agent.action_values[choice])
                
                if guess != None:   
                    agent.reset_channels()
                    correct += int(reward > 0)
            avg_reward/=1000.0   
            print "Iteration {} in environment with eps {}".format(i,env.eps)
            corrects.append(correct/10.0)
            lengths.append(length/1000.0)
            rewards.append(avg_reward)
        q.append(agent.action_values.copy())

    avg_reward = 0
    reward = None
    evidence = env.reset()       
    correct = 0
    length = 0
    channels = []
    while env.iters < 1000:
        length+=1
        choice = np.argmax(agent.action_values)
        agent.channels = agent.decay*agent.channels + agent.scales[choice]*evidence
        prob = softmax(agent.channels)
        guess = np.argmax(prob) if np.max(prob) > 0.3 else None
        reward,evidence = env.step(guess)
        avg_reward+=reward
        #agent.action_values[choice] += alpha*(reward + np.max(agent.action_values) - agent.action_values[choice])
        if guess != None:
            agent.reset_channels()
            correct += int(reward > 0)
        channels.append(agent.channels)
    avg_reward/=1000.0
    corrects.append(correct/10.0)
    lengths.append(length/1000.0)
    rewards.append(avg_reward)
    env.reset()
    print "eps = {} done in time {}. Final accuracy is {} %".format(env.eps,time.time() - start_time,correct/10.0)
    
    return lengths,corrects,rewards,agent
          
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='ArgMax Environment')
    parser.add_argument('--eps', type=float, default=None, metavar='E',
                        help='Spead of the distribution')
    args = parser.parse_args()
    
    if args.eps != None:
        #env = ArgmaxEnv(args.eps)
        #agent = Agent(env)
        #l,perf = train(agent,env)
        #q = agent.action_values
        #s = agent.scales[np.argmax(q)]
        #np.savez(str(args.eps)+'.npy',perf=perf,lengths=l,  scale=s)
        ret = train((args.eps,0))
        plt.figure()
        plt.plot(ret[0],label="eps = %.2f" % (args.eps))
        plt.figure()
        plt.plot(ret[1],label="eps = %.2f" % (args.eps))
        plt.figure()
        plt.plot(ret[2],label="eps = %.2f" % (args.eps))
        #plt.figure()
        #plt.imshow(np.array(ret[3]).T)
        #plt.figure()
        #plt.imshow(np.array(ret[-1]).T)
    else:
        p = Pool(10)
        ret = p.map(train,zip(np.arange(7,10)/10.0,np.ones((3,))*0.1))
#        scales = []
#        perfs = []
#        lengths = []
#        for i in range(10):
#            print i
#            env = ArgmaxEnv(i/10.0)
#            agent = Agent(env)
#            l,perf = train(agent,env)
#            perfs.append(perf)
#            lengths.append(l)
#            q = agent.action_values
#            scales.append(agent.scales[np.argmax(q)])
        
        lengths = [r[0] for r in ret]
        corrects = [r[1] for r in ret]
        rewards = [r[2] for r in ret]
        thresholds = [r[3].scales[np.argmax(r[3].action_values)] for r in ret]
    
        plt.figure()
        plt.plot(np.arange(7,10)/1.0,thresholds)
        
        plt.figure()
        for i in range(len(corrects)):
            plt.plot(corrects[i],label="eps = %.2f" % ((i+7)/10.0))
        plt.legend()
        
        plt.figure()
        for i in range(len(lengths)):
            plt.plot(lengths[i],label="eps = %.2f" % ((i+7)/10.0))
        plt.legend()
        
        plt.figure()
        for i in range(len(corrects)):
            plt.plot(rewards[i],label="eps = %.2f" % ((i+7)/10.0))
        plt.legend()
#        np.savez(str(args.eps)+'.npy',perfs=perfs,lengths=lengths,  scale=scales)