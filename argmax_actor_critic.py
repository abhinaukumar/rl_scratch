#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 11:35:28 2018

@author: nownow
"""

import gym
import numpy as np

import torch
from torch import nn
from torch.distributions import Categorical
from torch import optim
import torch.nn.functional as F

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

#class Policy(nn.Module):
#    def __init__(self):
#        super(Policy, self).__init__()
#        self.affine1 = nn.Linear(4, 128)
#        self.action_head = nn.Linear(128, 2)
#        self.value_head = nn.Linear(128, 1)
#
#        self.saved_actions = []
#        self.rewards = []
#
#    def forward(self, x):
#        x = F.relu(self.affine1(x))
#        action_scores = self.action_head(x)
#        state_values = self.value_head(x)
#        return F.softmax(action_scores, dim=-1), state_values

class ActorCritic(nn.Module):
    def __init__(self,input_dim,dims):
        super(ActorCritic,self).__init__()
        self.input_dim = input_dim
        #self.n_scales = 10
        #self.n_thresholds = 1
        self.n_actions = 10 #self.n_scales * self.n_thresholds
        self.layer1 = nn.Linear(input_dim,dims)
        self.action_layer = nn.Linear(dims,self.n_actions)
        self.value_layer = nn.Linear(dims,1)
        self.saved_log_probs = []
        self.rewards = []
        self.values = []
        
    def forward(self,state):
        h = nn.ReLU()(self.layer1(state))
        y1 = nn.Softmax(dim=-1)(self.action_layer(h))
        y2 = self.value_layer(h)
        return y1,y2
    
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
        self.model = ActorCritic(self.n_options,25).to(devices[self.eps_env])
        self.channels = np.zeros((self.n_options,))
        #self.scales = np.arange(1.0,11.0)
        #np.random.shuffle(self.scales)
        self.decay = 1.0
        self.losses = []
        #self.action_values = np.ones(self.n_actions,)
        #self.action_values = np.random.normal(1.0,0.25,(self.n_actions,))
                
    def reset_channels(self):
        self.channels = np.zeros((self.n_options,))
        
    def act(self,greedy=False):
        state = torch.from_numpy(self.channels).float().unsqueeze(0).to(devices[self.eps_env])
        probs,value = self.model.forward(state)
        #print probs
        if not greedy:
            #print probs
            dist = Categorical(probs)
            a = dist.sample()
            self.model.saved_log_probs.append(dist.log_prob(a))
            self.model.values.append(value)
            a = a.item()
        else:
            a = torch.argmax(probs).item()
            
        scale = 1 + a
        #print scale
        #threshold = (10 - self.model.n_thresholds + a%self.model.n_thresholds)/10.0
        return scale#,threshold
        
    def clear_memory(self):
        del self.model.saved_log_probs[:]
        del self.model.values[:]
            
def finish_episode(agent,optimizer):
    #policy_loss = []
#    rewards = []
#    R = 0
#    for r in agent.model.rewards[::-1]:
#        R = r + R
#        rewards.insert(0,R)
    #print agent.model.rewards[::-1],type(agent.model.rewards[::-1])
    #print np.cumsum(agent.model.rewards[::-1])[::-1]
    rewards = np.cumsum(agent.model.rewards[::-1])[::-1]
    rewards = torch.Tensor(rewards.copy()).to(devices[agent.eps_env])
    #std = 0 if rewards.shape == torch.Size([1]) else rewards.std()
    #rewards = (rewards - rewards.mean()) / (std + np_eps)
    #print rewards
    values = torch.squeeze(torch.cat(agent.model.values))
    #std = 0 if rewards.shape == torch.Size([1]) else rewards.std()
        
#    rewards = (rewards - rewards.mean()) / (std + np_eps)

#    for log_prob,reward in zip(agent.model.saved_log_probs,rewards):
#        policy_loss.append(-log_prob*reward)
#    R = 0
#    blah_rewards = []
#    policy_losses = []
#    value_losses = []
#    for r in agent.model.rewards[::-1]:
#        R = r + R
#        blah_rewards.insert(0, R)
#    blah_rewards = torch.tensor(blah_rewards).to(devices[agent.eps_env])
#    #blah_rewards = (blah_rewards - blah_rewards.mean()) / (blah_rewards.std() + np_eps)
#    for log_prob, value, r in zip(agent.model.saved_log_probs, agent.model.values, blah_rewards):
#        reward = r - value.item()
#        #print reward
#        policy_losses.append(-log_prob * reward)
#        value_losses.append(F.smooth_l1_loss(value, torch.tensor([[r]]).to(devices[agent.eps_env])))
#    optimizer.zero_grad()
#    blah_loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    optimizer.zero_grad()
    loss = torch.sum(-torch.cat(agent.model.saved_log_probs)*(rewards - values)) + nn.modules.loss.SmoothL1Loss(size_average=False)(values,rewards)
#    if loss != blah_loss:
#        print loss,blah_loss
    loss.backward()
    optimizer.step()
    
    agent.losses.append(loss.item())
    del agent.model.rewards[:]
    del agent.model.saved_log_probs[:]
    del agent.model.values[:]
    
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
        
    for i in range(250):
            
        # Report the performance of the greedy policy every 50 iterations
        
        tot_reward = 0
        reward = None
        evidence = env.reset()
        agent.reset_channels()
        correct = 0
        length = 0
        while env.iters < 500:
            length+=1
            scale = agent.act(greedy=True)
            #print scale,threshold
            agent.channels = agent.decay*agent.channels + scale*evidence
            prob = softmax(agent.channels)
            #print prob
            guess = np.argmax(prob) if np.max(prob) > 0.5 else None
            reward,evidence,done = env.step(guess)
            tot_reward+=reward
            
            if done:
                agent.reset_channels()
                correct += int(reward != -30)
                
        if not i%10:
            print "Iteration {} in environment with eps {}".format(i,env.eps)
            
        corrects.append(correct/5.0)
        lengths.append(length/500.0)
        rewards.append(tot_reward/500.0)
            #scales.append(agent.scales[np.argmax(agent.action_values)])
        agent.clear_memory()
        

        length = 0
        reward = None
        evidence = env.reset()
        agent.reset_channels()
        while env.iters < 250:
            #e = np.random.uniform()
            #choice = np.argmax(agent.action_values) if e >= eps else np.random.randint(0,agent.n_actions)
            scale = agent.act()
            agent.channels = agent.decay*agent.channels + scale*evidence # The scale chosen is a measure of the agent's confidence in the decision
            prob = softmax(agent.channels)
            guess = np.argmax(prob) if np.max(prob) > 0.5 else None
            reward,evidence,done = env.step(guess)
            agent.model.rewards.append(reward)
            if done:
                agent.reset_channels()
                finish_episode(agent,optimizer)

    tot_reward = 0
    reward = None
    evidence = env.reset()
    agent.reset_channels()
    correct = 0
    length = 0
    while env.iters < 500:
        length+=1
        scale = agent.act(greedy=True)
        agent.channels = agent.decay*agent.channels + scale*evidence
        prob = softmax(agent.channels)
        guess = np.argmax(prob) if np.max(prob) > 0.5 else None
        reward,evidence,done = env.step(guess)
        tot_reward+=reward
        
        if done:
            agent.reset_channels()
            correct += int(reward != -30)
        
    corrects.append(correct/5.0)
    lengths.append(length/500.0)
    rewards.append(tot_reward/500.0)
    #scales.append(agent.scales[np.argmax(agent.action_values)])
    agent.clear_memory()
    
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
            ret = train((args.eps,1e-3))
            plt.figure()
            plt.title('Delay vs time')
            plt.plot(ret[0],label="eps = %.2f" % (args.eps))
            plt.figure()
            plt.title('Accuracy vs time')
            plt.plot(ret[1],label="eps = %.2f" % (args.eps))
            plt.figure()
            plt.title('Reward vs time')
            plt.plot(ret[2],label="eps = %.2f" % (args.eps))
            plt.figure()
            plt.title('Loss vs time')
            plt.plot(ret[3].losses[::250],label="eps = %.2f" % (args.eps))
            
        else:
            p = Pool(10)
            ret = p.map(train,zip(np.arange(10)/10.0,np.ones((10,))*1e-2))
            
            lengths = [r[0] for r in ret]
            corrects = [r[1] for r in ret]
            rewards = [r[2] for r in ret]
            losses = [r[3].losses for r in ret]
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
            
            plt.figure()
            plt.title("Loss vs Time")
            for i in range(len(rewards)):
                plt.plot(losses[i][::250],label="eps = %.2f" % (i/10.0))
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
#            
#    else:
#        p = Pool(10)
#        ret = p.map(test,np.arange(10)/10.0)
#        
#        lengths = [r[0] for r in ret]
#        corrects = [r[1] for r in ret]
#        rewards = [r[2] for r in ret]
#    
#        plt.figure()
#        plt.title('Accuracy vs scale')
#        for i in range(len(corrects)):
#            plt.plot(np.arange(10)+1,corrects[i],label="eps = %.2f" % (i/10.0))
#        plt.legend()
#        
#        plt.figure()
#        plt.title('Delay vs scale')
#        for i in range(len(lengths)):
#            plt.plot(np.arange(10)+1,lengths[i],label="eps = %.2f" % (i/10.0))
#        plt.legend()
#        
#        plt.figure()
#        plt.title('Reward vs scale')
#        for i in range(len(rewards)):
#            plt.plot(np.arange(10)+1,rewards[i],label="eps = %.2f" % (i/10.0))
#        plt.legend()
        