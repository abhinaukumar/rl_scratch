# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:54:55 2018

@author: nownow
"""

import gym
import numpy as np

import torch
from torch import nn
from torch.distributions import Categorical
from torch import optim
from eligibility_trace_optimizer import EligibilityTraceSemiGradientTD

import argparse
from itertools import count

import matplotlib.pyplot as plt
plt.ion()

args = None
eps = np.finfo(np.float32).eps.item()

# Define the policy network

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

# Define the value network

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

# Act based on the current policy
        
def act(policy,state):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs = policy.forward(state)
    dist = Categorical(probs)
    a = dist.sample()
    policy.saved_log_probs.append(dist.log_prob(a))
    return a.item()
    
# Collect all rewards and actions of the episode
    
def finish_episode_reinforce(policy,optimizer):
    R = 0
    policy_loss = []
    rewards = []
    
    for r in policy.rewards[::-1]:
        R = r + R*args.gamma
        rewards.insert(0,R)
    rewards = torch.Tensor(rewards).to(device)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    
    for log_prob,reward in zip(policy.saved_log_probs,rewards):
        policy_loss.append(-log_prob*reward)
    
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum().to(device)
    policy_loss.backward()
    optimizer.step()
    
    del policy.rewards[:]
    del policy.saved_log_probs[:]

# REINFORCE - Monte Carlo policy gradient
    
def reinforce(policy,env):
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)    
    running_reward = 0.0
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
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(i, t, running_reward))
            lengths.append(t)
            
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break
            
    return lengths

# Carry out Semi gradient TD to evaluate the policy
    
def semi_gradient_td(env,policy,value,alpha,lamda,gamma):
    optimizer = EligibilityTraceSemiGradientTD(value.parameters(),alpha,decay=lamda*gamma)
    
    for i in range(100):
        state = env.reset()
        done = False
        steps = 0
        mean_delta = 0.0
        
        v = value.forward(torch.from_numpy(state).float().unsqueeze(0).to(device))
        
        optimizer.reset_traces()
        
        while not done:
            steps+=1
            v.backward()
            prob = policy.forward(torch.from_numpy(state).float().unsqueeze(0).to(device))
            act = np.argmax(prob.detach().cpu().numpy())
            state,reward,done,_ = env.step(act)
            v_next = value.forward(torch.from_numpy(state).float().unsqueeze(0).to(device))
            
            # Calculate TD error
            
            delta = reward + gamma*v - v_next
            optimizer.step(delta.item())
            #print list(value.parameters())
            v = v_next
            mean_delta = (mean_delta*(steps-1) + np.abs(delta.item()))/steps
        print "Average error at the end of episode {} is {}".format(i,mean_delta)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--lamda', type=float, default=0.10, metavar='L',
                        help='trace decay rate (default: 0.10)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    env = gym.make('CartPole-v0')
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    
    policy = Policy(4,128,2)
    value = Value(4,128)
    
    policy.to(device)
    value.to(device)
    
    lengths = reinforce(policy,env)
    semi_gradient_td(env,policy,value,1e-5,args.lamda,args.gamma)