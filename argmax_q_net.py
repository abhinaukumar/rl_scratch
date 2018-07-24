#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 11:00:57 2018

@author: nownow
"""

import numpy as np

import torch
from torch import nn
from torch import optim

import argparse
import time

from multiprocessing import Pool

import matplotlib.pyplot as plt
plt.ion()

args = None
np_eps = np.finfo(np.float32).eps.item()
devices = {}

def softmax(x):
    return np.exp(x)/np.expand_dims(np.sum(np.exp(x),axis=-1),axis=-1)

class DQN(nn.Module):
    def __init__(self,input_dim,dims):
        super(DQN,self).__init__()
        self.input_dim = input_dim
        self.n_scales = 5
        self.n_thresholds = 5
        self.n_actions = self.n_scales * self.n_thresholds
        self.layer1 = nn.Linear(input_dim,dims)
        self.action_layer = nn.Linear(dims,self.n_actions)
        self.rewards = []
        self.values = []
        self.next_q = []
        
    def forward(self,state):
        h = nn.ReLU()(self.layer1(state))
        y1 = self.action_layer(h)
        return y1
    
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
        self.dist = np.ones((self.n_options,))*self.eps/self.n_options
        self.dist[self.true_option] += 1 - self.eps
        
    # Step through the environment.
    def step(self,guess):
        self.prediction_time+=1
        done = False
        if guess != None:
            reward = 30.0 - self.prediction_time if guess == self.true_option else -30.0
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
        self.eps_env = env.eps
        self.model = DQN(self.n_options,25).to(devices[self.eps_env])
        self.channels = np.zeros((self.n_options,))
        self.decay = 1.0
        self.losses = []
                
    def reset_channels(self):
        self.channels = np.zeros((self.n_options,))
        
    def act(self,eps=0.0):
        state = torch.from_numpy(self.channels).float().unsqueeze(0).to(devices[self.eps_env])
        q = self.model.forward(state)
        e = np.random.uniform()
        if e > eps:
            a = 1 + torch.argmax(q).item()
        else:
            a = np.random.randint(1,11)
        self.model.values.append(q[:,a - 1])
        self.model.next_q.append(torch.max(q).unsqueeze(0))
            
        scale = 1 + a/self.model.n_thresholds
        threshold = (10 - self.model.n_thresholds + a%self.model.n_thresholds)/10.0
        return scale,threshold
        
    def clear_memory(self):
        del self.model.values[:]
        del self.model.next_q[:]
            
def finish_episode(agent,optimizer):
    agent.model.next_q.append(torch.Tensor([0]).to(devices[agent.eps_env]))
    rewards = torch.Tensor(agent.model.rewards).to(devices[agent.eps_env])
    values = torch.squeeze(torch.cat(agent.model.values))
    next_q = torch.squeeze(torch.cat(agent.model.next_q[1:]))

    optimizer.zero_grad()
    loss = torch.sum((rewards + next_q - values)**2)
    loss.backward()
    optimizer.step()
    
    agent.losses.append(loss.item())
    del agent.model.rewards[:]
    del agent.model.values[:]
    del agent.model.next_q[:]
    
def train(args): 
        
    eps_env,alpha = args
    devices[eps_env] = torch.device("cuda:" + str(int(eps_env*10)%1) if torch.cuda.is_available() else "cpu")
    
    start_time = time.time()
    env = ArgmaxEnv(eps_env)
    agent = Agent(env)
    
    optimizer = optim.Adam(agent.model.parameters(), lr=alpha)
    
    corrects = []
    lengths = []
    rewards = []
    eps = 1.0
    for i in range(100):
            
        # Report the performance of the greedy policy every 50 iterations
        eps *= 0.95
        reward = None
        evidence = env.reset()
        agent.reset_channels()
        correct = 0
        length = 0
        _rewards = []
        _lengths = []
        while env.iters < 500:
            length+=1
            scale,threshold = agent.act()
            agent.channels = agent.decay*agent.channels + scale*evidence
            prob = softmax(agent.channels)
            guess = np.argmax(prob) if np.max(prob) > threshold else None
            reward,evidence,done = env.step(guess)
            
            if done:
                _lengths.append(length)
                _rewards.append(reward)
                length = 0
                agent.reset_channels()
                correct += int(reward != -30)
                
        corrects.append(correct/5.0)
        lengths.append([np.mean(_lengths),np.mean(_lengths) - np.std(_lengths),np.mean(_lengths) + np.std(_lengths)])
        rewards.append([np.mean(_rewards),np.mean(_rewards) - np.std(_rewards),np.mean(_rewards) + np.std(_rewards)])
        agent.clear_memory()
        
        if not i%10:
            print "Iteration {} in environment with eps {}".format(i,env.eps)
            
        

        length = 0
        reward = None
        evidence = env.reset()
        agent.reset_channels()
        while env.iters < 500:
            scale,threshold = agent.act(eps)
            agent.channels = agent.decay*agent.channels + scale*evidence # The scale chosen is a measure of the agent's confidence in the decision
            prob = softmax(agent.channels)
            guess = np.argmax(prob) if np.max(prob) > threshold else None
            reward,evidence,done = env.step(guess)
            agent.model.rewards.append(reward)
            if done:
                agent.reset_channels()
                finish_episode(agent,optimizer)

    reward = None
    evidence = env.reset()
    agent.reset_channels()
    correct = 0
    length = 0
    _rewards = []
    _lengths = []
    while env.iters < 500:
        length+=1
        scale,threshold = agent.act()
        agent.channels = agent.decay*agent.channels + scale*evidence
        prob = softmax(agent.channels)
        guess = np.argmax(prob) if np.max(prob) > threshold else None
        reward,evidence,done = env.step(guess)
        
        if done:
            _rewards.append(reward)
            _lengths.append(length)
            length = 0
            agent.reset_channels()
            correct += int(reward != -30)
        
    corrects.append(correct/5.0)
    lengths.append([np.mean(_lengths),np.mean(_lengths) - np.std(_lengths),np.mean(_lengths) + np.std(_lengths)])
    rewards.append([np.mean(_rewards),np.mean(_rewards) - np.std(_rewards),np.mean(_rewards) + np.std(_rewards)])
    agent.clear_memory()
    
    print "eps = {} done in time {}. Final accuracy is {}%, average decision time is {} and average reward is {}".format(env.eps,time.time() - start_time,corrects[-1], lengths[-1], rewards[-1])
    
    return lengths,corrects,rewards,agent

if __name__ == '__main__':
    
    np.random.seed(0)
    torch.manual_seed(0)
    
    parser = argparse.ArgumentParser(description='ArgMax Environment')
    parser.add_argument('--test', action='store_true',
                        help='Test')
    parser.add_argument('--eps', type=float, default=None, metavar='E',
                        help='Spead of the distribution')
    args = parser.parse_args()
    if not args.test:
        if args.eps != None:
            ret = train((args.eps,1e-3))
            plt.figure()
            plt.title('Delay vs time')
            plt.plot(np.array(ret[0])[:,0],label="eps = %.2f" % (args.eps))
            plt.fill_between(np.arange(len(ret[0])),np.array(ret[0])[:,1], np.array(ret[0])[:,2],alpha=0.5)
            plt.figure()
            plt.title('Accuracy vs time')
            plt.plot(ret[1],label="eps = %.2f" % (args.eps))
            plt.figure()
            plt.title('Reward vs time')
            plt.plot(np.array(ret[2])[:,0],label="eps = %.2f" % (args.eps))
            plt.fill_between(np.arange(len(ret[2])),np.array(ret[2])[:,1], np.array(ret[2])[:,2],alpha=0.5)
            plt.figure()
            plt.title('Loss vs time')
            plt.plot(ret[3].losses[::250],label="eps = %.2f" % (args.eps))
            
        else:
            p = Pool(10)
            ret = p.map(train,zip(np.arange(5)/5.0,np.ones((5,))*1e-3))
            lengths = [np.array(r[0])[:,0] for r in ret]
            corrects = [r[1] for r in ret]
            rewards = [np.array(r[2])[:,0] for r in ret]
            
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