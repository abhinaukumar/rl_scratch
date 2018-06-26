#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 13:25:21 2018

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
    def __init__(self,input_dim,hidden_dim,n_actions,device):
        super(DQN,self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.layer1 = nn.Linear(input_dim,self.hidden_dim)
        self.gru = nn.GRUCell(self.hidden_dim,self.hidden_dim/2)
        self.action_layer = nn.Linear(self.hidden_dim/2,n_actions+1)
        self.rewards = []
        self.values = []
        self.ht = torch.zeros((1,self.hidden_dim/2)).to(self.device)
        self.next_q = []
        
    def forward(self,state):
        h = nn.ReLU()(self.layer1(state))
        self.ht = self.gru(h,self.ht)
        
        y1 = self.action_layer(self.ht)
        return y1
    
    def reset(self):
        
        del self.rewards[:]
        del self.values[:]
        del self.next_q[:]
    
        self.ht = self.ht.new_zeros(self.ht.size())# torch.zeros((1,self.hidden_dim/2)).to(self.device)
        
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
        self.device = devices[self.eps_env]
        self.model = DQN(self.n_options,25,self.n_options,self.device).to(self.device)
        self.losses = []
        self.q = []
        
    def reset_channels(self):
        self.channels = np.zeros((self.n_options,))
        
    def act(self,evidence,eps=0.0):
        state = torch.from_numpy(evidence).float().unsqueeze(0).to(self.device)
        q = self.model.forward(state)
        if eps!=0.0:
            self.q.append(q)
        e = np.random.uniform()
        if e > eps:
            a = torch.argmax(q).item()
        else:
            a = np.random.randint(0,self.n_options + 1)
            
        self.model.values.append(q[:,a])
        self.model.next_q.append(torch.max(q).unsqueeze(0))
        
        return a if a < self.n_options else None
        
def finish_episode(agent,optimizer):
    agent.model.next_q.append(torch.Tensor([0]).to(agent.device))
    rewards = torch.Tensor(agent.model.rewards).to(agent.device)
    values = torch.squeeze(torch.cat(agent.model.values))
    next_q = torch.squeeze(torch.cat(agent.model.next_q[1:]))

    optimizer.zero_grad()
    loss = nn.modules.MSELoss()(values,rewards + next_q.detach())
    loss.backward()
    optimizer.step()
    
    agent.losses.append(loss.item())
    del agent.model.rewards[:]
    del agent.model.values[:]
    del agent.model.next_q[:]
    
def train(args): 
        
    eps_env,alpha = args
    devices[eps_env] = torch.device("cuda:" + str(int(eps_env*10)%1))
    
    start_time = time.time()
    env = ArgmaxEnv(eps_env)
    agent = Agent(env)
    
    optimizer = optim.Adam(agent.model.parameters(), lr=alpha)
    
    corrects = []
    lengths = []
    rewards = []
    
    eps = 1.0
    for i in range(150):
        
        eps*=0.95
        # Report the performance of the greedy policy every 50 iterations
        
        tot_reward = 0
        reward = None
        evidence = env.reset()
        agent.reset_channels()
        correct = 0
        length = 0
        
        _rewards = []
        _lengths = []
        while env.iters < 500:
            length+=1
            guess = agent.act(evidence)
            reward,evidence,done = env.step(guess)
            
            if done:
                agent.reset_channels()
                _rewards.append(reward)
                _lengths.append(length)
                length = 0
                correct += int(reward != -30)
                agent.model.reset()
                
        corrects.append(correct/5.0)
        lengths.append([np.mean(_lengths),np.mean(_lengths) - np.std(_lengths),np.mean(_lengths) + np.std(_lengths)])
        rewards.append([np.mean(_rewards),np.mean(_rewards) - np.std(_rewards),np.mean(_rewards) + np.std(_rewards)])
        
        if not i%10:
            print "Iteration {} in environment with eps {}".format(i,env.eps)
            print "{}, {}, {}".format(corrects[-1],lengths[-1],rewards[-1])
        
        length = 0
        reward = None
        evidence = env.reset()
        agent.reset_channels()
        while env.iters < 500:
            guess = agent.act(evidence,eps)
            reward,evidence,done = env.step(guess)
            agent.model.rewards.append(reward)
            if done:
                agent.reset_channels()
                finish_episode(agent,optimizer)
                agent.model.reset()

    tot_reward = 0
    reward = None
    evidence = env.reset()
    agent.reset_channels()
    correct = 0
    length = 0
    _rewards = []
    _lengths = []
    while env.iters < 500:
        length+=1
        guess = agent.act(evidence)
        reward,evidence,done = env.step(guess)
        tot_reward+=reward
        
        if done:
            agent.reset_channels()
            _lengths.append(length)
            length = 0
            _rewards.append(reward)
            correct += int(reward != -30)
            agent.model.reset()
        
    corrects.append(correct/5.0)
    lengths.append([np.mean(_lengths),np.mean(_lengths) - np.std(_lengths),np.mean(_lengths) + np.std(_lengths)])
    rewards.append([np.mean(_rewards),np.mean(_rewards) - np.std(_rewards),np.mean(_rewards) + np.std(_rewards)])
        
    agent.device = agent.model.device = None
    print "eps = {} done in time {}. Final accuracy is {}%, average decision time is {} and average reward is {}".format(env.eps,time.time() - start_time,corrects[-1], lengths[-1], rewards[-1])
    
    return lengths,corrects,rewards,agent

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='ArgMax Environment')
    parser.add_argument('--test', action='store_true',
                        help='Test')
    parser.add_argument('--eps', type=float, default=None, metavar='E',
                        help='Spead of the distribution')
    args = parser.parse_args()
    if not args.test:
        if args.eps != None:
            ret = train((args.eps,3e-3))
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
            ret = p.map(train,zip(np.arange(10)/10.0,np.ones((10,))*3e-3))
            lengths = [np.array(r[0])[:,0] for r in ret]
            l_min = [np.array(r[0])[:,1] for r in ret]
            l_max = [np.array(r[0])[:,2] for r in ret]
            corrects = [r[1] for r in ret]
            rewards = [np.array(r[2])[:,0] for r in ret]
            r_min = [np.array(r[2])[:,1] for r in ret]
            r_max = [np.array(r[2])[:,2] for r in ret]
            losses = [r[3].losses[::250] for r in ret]
            
            plt.figure()
            plt.title("Accuracy vs Time")
            for i in range(len(corrects)):
                plt.plot(corrects[i],label="eps = %.2f" % (i/10.0))
            plt.legend()
            
            plt.figure()
            plt.title("Delay vs Time")
            for i in range(len(lengths)):
                plt.plot(lengths[i],label="eps = %.2f" % (i/10.0))
                plt.fill_between(np.arange(len(lengths[i])), l_min[i], l_max[i], alpha=0.5)
            plt.legend()
            
            plt.figure()
            plt.title("Reward vs Time")
            for i in range(len(rewards)):
                plt.plot(rewards[i],label="eps = %.2f" % (i/10.0))
                plt.fill_between(np.arange(len(rewards[i])), r_min[i], r_max[i], alpha=0.5)
            plt.legend()
            
            plt.figure()
            plt.title("Loss vs Time")
            for i in range(len(rewards)):
                plt.plot(losses[i],label="eps = %.2f" % (i/10.0))
            plt.legend()