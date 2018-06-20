#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:53:41 2018

@author: nownow
"""

import gym
import numpy as np

import torch
from torch import nn
from torch.distributions import Categorical
from torch import optim

import argparse
from itertools import count

import matplotlib.pyplot as plt
plt.ion()

import time

from multiprocessing import Pool

args = None
eps = np.finfo(np.float32).eps.item()
devices = {}

def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x))

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

class EvidenceExtractor(nn.Module):
    def __init__(self,input_dim,dims,n_actions):
        super(EvidenceExtractor,self).__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.layer1 = nn.Linear(input_dim,dims)
        self.layer2 = nn.Linear(dims,n_actions)
        
    def forward(self,state):
        h = nn.ReLU()(self.layer1(state))
        y = self.layer2(h)
        return y

class Agent():
    def __init__(self,env):
        self.n_options = env.n_options
        self.eps_env = env.eps
        self.n_actions = 10
        #self.policy = Policy(self.n_options,25,self.n_actions).to(devices[self.eps_env])
        #self.value = Value(self.n_options,25).to(devices[self.eps_env])
        self.model = EvidenceExtractor(self.n_options,25,self.n_actions).to(devices[self.eps_env])
        self.channels = torch.zeros((self.n_options,)).to(devices[self.eps_env])
        self.scales = np.arange(1.0,11.0)
        np.random.shuffle(self.scales)
        self.decay = 1.0
        #self.action_values = np.ones(self.n_actions,)
        #self.action_values = np.random.normal(1.0,0.25,(self.n_actions,))
                
    def reset_channels(self):
        self.channels = torch.zeros((self.n_options,)).to(devices[self.eps_env])
        
#    def act(self,greedy=False):
#        state = self.channels.to(devices[self.eps_env])
#        probs = self.policy.forward(state)
##        print probs
#        if not greedy:
#            dist = Categorical(probs)
#            a = dist.sample()
#            self.policy.saved_log_probs.append(dist.log_prob(a))
#            return a.item()
#        else:
#            return torch.argmax(probs).item()
    

def train(args): 
        
    eps_env,alpha = args
    devices[eps_env] = torch.device("cuda:" + str(int(eps_env*10)%8) if torch.cuda.is_available() else "cpu")
    
    start_time = time.time()
    env = ArgmaxEnv(eps_env)
    agent = Agent(env)
    
    optimizer = optim.Adam(agent.model.parameters(), lr=alpha)
    
    corrects = []
    lengths = []
    rewards = []
    #p = []
    #scales = []
        
    for i in range(100):
            
        # Report the performance of the greedy policy every 50 iterations
        
        tot_reward = 0
        reward = None
        evidence = env.reset()
        agent.reset_channels()
        correct = 0
        length = 0
        while env.iters < 1000:
            length+=1
            #choice = agent.act()
            agent.channels = agent.decay*agent.channels + agent.model.forward(torch.Tensor(evidence).to(devices[eps_env]))
            prob = softmax(agent.channels)
            guess = torch.argmax(prob).item() if torch.max(prob) > 0.5 else None
            reward,evidence,done = env.step(guess)
            tot_reward+=reward
            
            if done:                
                agent.reset_channels()
                correct += int(reward != -30)
            
        corrects.append(correct/10.0)
        lengths.append(length/1000.0)
        rewards.append(tot_reward/1000.0)
        #scales.append(agent.scales[np.argmax(agent.action_values)])
        #agent.clear_memory()
        
        if not i%10:
            print "Iteration {} in environment with eps {}".format(i,env.eps)
            print corrects[-1],lengths[-1],rewards[-1]

        length = 0
        reward = None
        evidence = env.reset()
        agent.reset_channels()
        while env.iters < 50:
            length +=1 
            #e = np.random.uniform()
            #choice = np.argmax(agent.action_values) if e >= eps else np.random.randint(0,agent.n_actions)
            #choice = agent.act()
            agent.channels = agent.decay*agent.channels + agent.model.forward(torch.Tensor(evidence).to(devices[eps_env])) # The scale chosen is a measure of the agent's confidence in the decision
            prob = softmax(agent.channels)
            guess = torch.argmax(prob).item() if torch.max(prob) > 0.5 else None
            reward,evidence,done = env.step(guess)
            #agent.policy.rewards.append(reward)
            if done:
                target = np.zeros((agent.n_options,))
                target[guess] = reward
                #target = softmax(torch.Tensor(target).to(devices[eps_env]))
                target = torch.Tensor(target).to(devices[eps_env])
                #print prob,target
                loss = nn.modules.loss.BCELoss()(prob,target)
                loss.backward()
                optimizer.step()
                
                agent.reset_channels()
                #print length
                length = 0
                #finish_episode(agent,optimizer)

    tot_reward = 0
    reward = None
    evidence = env.reset()
    agent.reset_channels()
    correct = 0
    length = 0
    while env.iters < 1000:
            length+=1
            #choice = agent.act()
            agent.channels = agent.decay*agent.channels + agent.model.forward(torch.Tensor(evidence).to(devices[eps_env]))
            prob = softmax(agent.channels)
            guess = torch.argmax(prob).item() if torch.max(prob) > 0.5 else None
            reward,evidence,done = env.step(guess)
            tot_reward+=reward
            
            if done:                
                agent.reset_channels()
                correct += int(reward != -30)
        
    corrects.append(correct/10.0)
    lengths.append(length/1000.0)
    rewards.append(tot_reward/1000.0)
    #scales.append(agent.scales[np.argmax(agent.action_values)])
    #agent.clear_memory()
    
    print "eps = {} done in time {}. Final accuracy is {}%, average decision time is {} and average reward is {}".format(env.eps,time.time() - start_time,corrects[-1], lengths[-1], rewards[-1])
    
    return lengths,corrects,rewards,agent

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='ArgMax Environment')
    parser.add_argument('--test', action='store_true',
                        help='Test')
    parser.add_argument('--eps', type=float, default=None, metavar='E',
                        help='Spead of the distribution')
    args = parser.parse_args()
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not args.test:
        if args.eps != None:
            ret = train((args.eps,1e-4))
            plt.figure()
            plt.title('Delay vs time')
            plt.plot(ret[0],label="eps = %.2f" % (args.eps))
            plt.figure()
            plt.title('Accuracy vs time')
            plt.plot(ret[1],label="eps = %.2f" % (args.eps))
            plt.figure()
            plt.title('Reward vs time')
            plt.plot(ret[2],label="eps = %.2f" % (args.eps))
            
        else:
            p = Pool(10)
            ret = p.map(train,zip(np.arange(10)/10.0,np.ones((10,))*1e-4))
            
            lengths = [r[0] for r in ret]
            corrects = [r[1] for r in ret]
            rewards = [r[2] for r in ret]
#            scales = [r[3] for r in ret]
#            q = [softmax(np.array(r[-1]),axis=-1).T for r in ret]
            
            plt.figure()
            plt.title("Accuracy vs Time")
            for i in range(len(corrects)):
                plt.plot(corrects[i],label="eps = %.2f" % (i/10.0))
            plt.legend()
            
            plt.figure()
            plt.title("Delay vs Time")
            for i in range(len(lengths)):
                plt.plot(lengths[i],label="eps = %.2f" % (i/10.0))
            plt.legend()
            
            plt.figure()
            plt.title("Reward vs Time")
            for i in range(len(rewards)):
                plt.plot(rewards[i],label="eps = %.2f" % (i/10.0))
            plt.legend()