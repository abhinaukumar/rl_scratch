#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 13:05:19 2018

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
import time

from multiprocessing import Pool

import matplotlib.pyplot as plt
plt.ion()

args = None
np_eps = np.finfo(np.float32).eps.item()
devices = {}

def softmax(x):
    return np.exp(x)/np.expand_dims(np.sum(np.exp(x),axis=-1),axis=-1)

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

class PolicyShared(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values
 
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
        self.eps_env = env.eps
        self.n_actions = 10
        self.policy = Policy(self.n_options,25,self.n_actions).to(devices[self.eps_env])
        self.value = Value(self.n_options,25).to(devices[self.eps_env])
        self.channels = np.zeros((self.n_options,))
        self.scales = np.arange(1.0,11.0)
        np.random.shuffle(self.scales)
        self.decay = 1.0
        #self.action_values = np.ones(self.n_actions,)
        #self.action_values = np.random.normal(1.0,0.25,(self.n_actions,))
                
    def reset_channels(self):
        self.channels = np.zeros((self.n_options,))
        
    def act(self,greedy=False):
        state = torch.from_numpy(self.channels).float().unsqueeze(0).to(devices[self.eps_env])
        probs = self.policy.forward(state)
#        print probs
        if not greedy:
            dist = Categorical(probs)
            a = dist.sample()
            self.policy.saved_log_probs.append(dist.log_prob(a))
            return a.item()
        else:
            return torch.argmax(probs).item()


def finish_episode_reinforce(agent,optimizer):
    policy_loss = []
    rewards = []
    
    #for r in agent.policy.rewards[::-1]:
    #    R = r + R
    #    rewards.insert(0,R)
    rewards = np.cumsum(agent.policy.rewards)
    rewards = torch.Tensor(rewards).to(devices[agent.eps_env])
    std = 0 if rewards.shape == torch.Size([1]) else rewards.std()
        
    rewards = (rewards - rewards.mean()) / (std + np_eps)
    
    for log_prob,reward in zip(agent.policy.saved_log_probs,rewards):
        policy_loss.append(-log_prob*reward)
    
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum().to(devices[agent.eps_env])
    policy_loss.backward()
    optimizer.step()
    
    del agent.policy.rewards[:]
    del agent.policy.saved_log_probs[:]

def finish_episode_reinforce_with_baseline(agent,policy_optimizer,baseline_optimizer):
    #R = 0
    policy_loss = []
    baseline_loss = []
    #rewards = []
    #deltas = []
    #for (r,v) in zip(policy.rewards[::-1],value.values[::-1]):
    #    R = r + R*args.gamma
    #    #rewards.insert(0,R)
    #    deltas.insert(0,R - v)
    #print agent.policy.rewards
    #print agent.value.values
    #print len(agent.policy.rewards)
    rewards = torch.Tensor(np.cumsum(agent.policy.rewards))
    values = torch.Tensor(agent.value.values)
    
    #print rewards.shape,values.shape
    
    deltas = rewards - values
    #print agent.value.values[0]
    deltas = torch.Tensor(deltas).to(devices[agent.eps_env])
    #rewards = torch.Tensor(rewards)
    #rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    #std = 0 if deltas.shape == torch.Size([1]) else deltas.std()
    #deltas = (deltas - deltas.mean()) / (std + np_eps)
    
    for log_prob,v,delta in zip(agent.policy.saved_log_probs,agent.value.values,deltas):
        policy_loss.append(-log_prob*delta)
        baseline_loss.append(-v*delta)

    policy_optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum().to(devices[agent.eps_env])
    policy_loss.backward()
    policy_optimizer.step()
    
    baseline_optimizer.zero_grad()
    baseline_loss = torch.cat(baseline_loss).sum().to(devices[agent.eps_env])
    baseline_loss.backward()
    baseline_optimizer.step()
    
    del agent.policy.rewards[:]
    del agent.policy.saved_log_probs[:]
    del agent.value.values[:]
    
def train(args): 
        
    eps_env,alpha = args
    devices[eps_env] = torch.device("cuda:" + str(int(eps_env*10)%8) if torch.cuda.is_available() else "cpu")
    
    start_time = time.time()
    env = ArgmaxEnv(eps_env)
    agent = Agent(env)
    
    optimizer = optim.Adam(agent.policy.parameters(), lr=alpha)
    
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
            choice = agent.act()
            agent.channels = agent.decay*agent.channels + agent.scales[choice]*evidence
            prob = softmax(agent.channels)
            guess = np.argmax(prob) if np.max(prob) > 0.5 else None
            reward,evidence,done = env.step(guess)
            tot_reward+=reward
            
            if done:
                agent.reset_channels()
                correct += int(reward != -30)
                
        if not i%10:
            print "Iteration {} in environment with eps {}".format(i,env.eps)
            
        corrects.append(correct/10.0)
        lengths.append(length/1000.0)
        rewards.append(tot_reward/1000.0)
            #scales.append(agent.scales[np.argmax(agent.action_values)])
            
        

        length = 0
        reward = None
        evidence = env.reset()
        agent.reset_channels()
        while env.iters < 500:
            #e = np.random.uniform()
            #choice = np.argmax(agent.action_values) if e >= eps else np.random.randint(0,agent.n_actions)
            choice = agent.act()
            agent.channels = agent.decay*agent.channels + agent.scales[choice]*evidence # The scale chosen is a measure of the agent's confidence in the decision
            prob = softmax(agent.channels)
            guess = np.argmax(prob) if np.max(prob) > 0.5 else None
            reward,evidence,done = env.step(guess)
            agent.policy.rewards.append(reward)
            if done:
                agent.reset_channels()
                finish_episode_reinforce(agent,optimizer)

    tot_reward = 0
    reward = None
    evidence = env.reset()
    agent.reset_channels()
    correct = 0
    length = 0
    while env.iters < 1000:
        length+=1
        choice = agent.act()
        agent.channels = agent.decay*agent.channels + agent.scales[choice]*evidence
        prob = softmax(agent.channels)
        guess = np.argmax(prob) if np.max(prob) > 0.5 else None
        reward,evidence,done = env.step(guess)
        tot_reward+=reward
        
        if done:
            agent.reset_channels()
            correct += int(reward != -30)
        
    corrects.append(correct/10.0)
    lengths.append(length/1000.0)
    rewards.append(tot_reward/1000.0)
    #scales.append(agent.scales[np.argmax(agent.action_values)])
            
    print "eps = {} done in time {}. Final accuracy is {}%, average decision time is {} and average reward is {}".format(env.eps,time.time() - start_time,corrects[-1], lengths[-1], rewards[-1])
    
    return lengths,corrects,rewards,agent

def train_with_baseline(args): 
        
    eps_env,alpha = args
    devices[eps_env] = torch.device("cuda:" + str(int(eps_env*10)%8) if torch.cuda.is_available() else "cpu")
    
    start_time = time.time()
    env = ArgmaxEnv(eps_env)
    agent = Agent(env)
    
    policy_optimizer = optim.Adam(agent.policy.parameters(), lr=alpha)
    baseline_optimizer = optim.Adam(agent.value.parameters(), lr=alpha)
    
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
            #agent.value.values.append(agent.value.forward(torch.from_numpy(agent.channels).float().unsqueeze(0).to(devices[eps_env])))  
            #print len(agent.value.values)
            choice = agent.act()
            agent.channels = agent.decay*agent.channels + agent.scales[choice]*evidence
            prob = softmax(agent.channels)
            guess = np.argmax(prob) if np.max(prob) > 0.5 else None
            reward,evidence,done = env.step(guess)
            tot_reward+=reward
            
            if done:
                agent.reset_channels()
                correct += int(reward != -30)
                
        if not i%10:
            print "Iteration {} in environment with eps {}".format(i,env.eps)
            
        corrects.append(correct/10.0)
        lengths.append(length/1000.0)
        rewards.append(tot_reward/1000.0)
            #scales.append(agent.scales[np.argmax(agent.action_values)])
            
        

        length = 0
        reward = None
        evidence = env.reset()
        agent.reset_channels()
        while env.iters < 500:
            #e = np.random.uniform()
            #choice = np.argmax(agent.action_values) if e >= eps else np.random.randint(0,agent.n_actions)
            agent.value.values.append(agent.value.forward(torch.from_numpy(agent.channels).float().unsqueeze(0).to(devices[eps_env])))
            choice = agent.act()
            agent.channels = agent.decay*agent.channels + agent.scales[choice]*evidence # The scale chosen is a measure of the agent's confidence in the decision
            prob = softmax(agent.channels)
            guess = np.argmax(prob) if np.max(prob) > 0.5 else None
            reward,evidence,done = env.step(guess)
            agent.policy.rewards.append(reward)
            if done:
                agent.reset_channels()
                finish_episode_reinforce_with_baseline(agent,policy_optimizer,baseline_optimizer)

    tot_reward = 0
    reward = None
    evidence = env.reset()
    agent.reset_channels()
    correct = 0
    length = 0
    while env.iters < 1000:
        length+=1
        choice = agent.act()
        agent.channels = agent.decay*agent.channels + agent.scales[choice]*evidence
        prob = softmax(agent.channels)
        guess = np.argmax(prob) if np.max(prob) > 0.5 else None
        reward,evidence,done = env.step(guess)
        tot_reward+=reward
        
        if done:
            agent.reset_channels()
            correct += int(reward != -30)
        
    corrects.append(correct/10.0)
    lengths.append(length/1000.0)
    rewards.append(tot_reward/1000.0)
    #scales.append(agent.scales[np.argmax(agent.action_values)])
            
    print "eps = {} done in time {}. Final accuracy is {}%, average decision time is {} and average reward is {}".format(env.eps,time.time() - start_time,corrects[-1], lengths[-1], rewards[-1])
    
    return lengths,corrects,rewards,agent

def test(env_eps):
        
    env = ArgmaxEnv(env_eps)
    agent = Agent(env)
    corrects = []
    rewards = []
    lengths = []
    
    for i in range(10):   
        tot_reward = 0
        reward = None
        evidence = env.reset()
        agent.reset_channels()
        correct = 0
        length=0
        while env.iters < 1000:
            length+=1
            choice = agent.act(greedy=True)
            agent.channels = agent.decay*agent.channels + agent.scales[choice]*evidence
            prob = softmax(agent.channels)
            guess = np.argmax(prob) if np.max(prob) > 0.5 else None
            reward,evidence,done = env.step(guess)
            tot_reward+=reward
            
            if done:
                agent.reset_channels()
                correct += int(reward != -30)
            
        corrects.append(correct/50.0)
        lengths.append(length/5000.0)
        rewards.append(tot_reward/5000.0)
        env.reset()
        
    return lengths,corrects,rewards

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
            ret = train_with_baseline((args.eps,3e-4))
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
            ret = p.map(train_with_baseline,zip(np.arange(10)/10.0,np.ones((10,))*3e-4))
            
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
            
#            fig,axes = plt.subplots(len(q),sharex=True)
#            for i,ax in enumerate(axes):
#                ax.set_title("eps = %.2f" % (i/10.0))
#                ax.imshow(q[i])
#                
#            fig,axes = plt.subplots(len(scales),sharex=True)
#            for i,ax in enumerate(axes):
#                ax.set_title("eps = %.2f" % (i/10.0))
#                ax.plot(scales[i])
            
    else:
        p = Pool(10)
        ret = p.map(test,np.arange(10)/10.0)
        
        lengths = [r[0] for r in ret]
        corrects = [r[1] for r in ret]
        rewards = [r[2] for r in ret]
    
        plt.figure()
        plt.title('Accuracy vs scale')
        for i in range(len(corrects)):
            plt.plot(np.arange(10)+1,corrects[i],label="eps = %.2f" % (i/10.0))
        plt.legend()
        
        plt.figure()
        plt.title('Delay vs scale')
        for i in range(len(lengths)):
            plt.plot(np.arange(10)+1,lengths[i],label="eps = %.2f" % (i/10.0))
        plt.legend()
        
        plt.figure()
        plt.title('Reward vs scale')
        for i in range(len(rewards)):
            plt.plot(np.arange(10)+1,rewards[i],label="eps = %.2f" % (i/10.0))
        plt.legend()