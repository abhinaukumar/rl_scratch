# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 13:25:47 2018

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

args = None
eps = np.finfo(np.float32).eps.item()

class Policy(nn.Module):
    def __init__(self,input_dim,dims,n_actions):
        super(Policy,self).__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.layer1 = nn.Linear(input_dim,dims)
        self.layer2 = nn.Linear(dims,n_actions)
        self.saved_log_probs = []
        self.rewards = []
        
    def forward(self,state):
        h = nn.ReLU()(self.layer1(state))
        y = nn.Softmax()(self.layer2(h))  
        return y
        
class Value(nn.Module):
    def __init__(self,input_dim,dims):
        super(Value,self).__init__()
        self.input_dim = input_dim
        self.layer1 = nn.Linear(input_dim,dims)
        self.layer2 = nn.Linear(dims,1)
        self.values = []
        
    def forward(self,state):
        h = nn.ReLU()(self.layer1(state))
        y = self.layer2(h)
        return y
        
def act(policy,state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy.forward(state)
    dist = Categorical(probs)
    a = dist.sample()
    policy.saved_log_probs.append(dist.log_prob(a))
    return a.item()

def finish_episode_reinforce(policy,optimizer):
    R = 0
    policy_loss = []
    rewards = []
    
    for r in policy.rewards[::-1]:
        R = r + R*args.gamma
        rewards.insert(0,R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    
    for log_prob,reward in zip(policy.saved_log_probs,rewards):
        policy_loss.append(-log_prob*reward)
    
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    
def reinforce(policy,env):
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)    
    running_reward = 10
    lengths = []
    
    for i in count(1):
        state = env.reset()
        for t in range(10000):
            action = act(policy,state)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            if args.render:
                env.render()
            if done:
                break
        running_reward = running_reward * 0.99 + t * 0.01
        
        finish_episode_reinforce(policy,optimizer)
        
        if i % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i, t, running_reward))
            lengths.append(t)
            
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break
            
    return lengths
    
def finish_episode_reinforce_with_baseline(policy,value,policy_optimizer,baseline_optimizer):
    R = 0
    policy_loss = []
    baseline_loss = []
    #rewards = []
    deltas = []
    print torch.Tensor(policy.rewards).shape,torch.Tensor(value.values).shape
    
    for (r,v) in zip(policy.rewards[::-1],value.values[::-1]):
        R = r + R*args.gamma
        #rewards.insert(0,R)
        deltas.insert(0,R - v)
    #rewards = torch.Tensor(rewards)
    #rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    
    deltas = torch.Tensor(deltas)
    deltas = (deltas - deltas.mean()) / (deltas.std() + eps)
    
    for log_prob,v,delta in zip(policy.saved_log_probs,value.values,deltas):
        policy_loss.append(-log_prob*delta)
        baseline_loss.append(-v*delta)

    policy_optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    policy_optimizer.step()
    
    baseline_optimizer.zero_grad()
    baseline_loss = torch.cat(baseline_loss).sum()
    baseline_loss.backward()
    baseline_optimizer.step()
    
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    del value.values[:]
    
def reinforce_with_baseline(policy,value,env):
    policy_optimizer = optim.Adam(policy.parameters(), lr=1e-2) 
    baseline_optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    running_reward = 10
    lengths = []
    
    for i in count(1):
        state = env.reset()
        for t in range(10000):
            action = act(policy,state)
            value.values.append(value.forward(torch.from_numpy(state).float().unsqueeze(0)))
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            if args.render:
                env.render()
            if done:
                break
        running_reward = running_reward * 0.99 + t * 0.01
        
        finish_episode_reinforce_with_baseline(policy,value,policy_optimizer,baseline_optimizer)
        
        if i % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i, t, running_reward))
            lengths.append(t)
            
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break
    return lengths
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--lambda', type=float, default=0.90, metavar='L',
                        help='trace decay rate (default: 0.90)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    
    args = parser.parse_args()
    
    env = gym.make('CartPole-v0')
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    
    policy1 = Policy(4,128,2)
    value = Value(4,128)
    lengths = reinforce_with_baseline(policy1,value,env)
    
    plt.figure()
    plt.plot(np.arange(1,args.log_interval*len(lengths)+1,args.log_interval),lengths,label='With baseline')
    
    policy2 = Policy(4,128,2)
    value = Value(4,128)
    lengths = reinforce(policy2,env)

    plt.plot(np.arange(1,args.log_interval*len(lengths)+1,args.log_interval),lengths,label='Without baseline')
    plt.legend()
