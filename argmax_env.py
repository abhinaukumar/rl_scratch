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


def train(args):
    eps_env,alpha = args
    start_time = time.time()
    env = ArgmaxEnv(eps_env)
    agent = Agent(env)
    
    corrects = []
    lengths = []
    rewards = []
    eps = 0.2
    q = []
    scales = []
        
    for i in range(1000):
        
        if not i%5:
            
            # Report the performance of the greedy policy every 50 iterations
            
            tot_reward = 0
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
                tot_reward+=reward
                
                if done:
                    agent.reset_channels()
                    correct += int(reward != -30)
                    
            if not i%50:
                print "Iteration {} in environment with eps {}".format(i,env.eps)
                
            corrects.append(correct/10.0)
            lengths.append(length/1000.0)
            rewards.append(tot_reward/1000.0)
            q.append(agent.action_values.copy())
            scales.append(agent.scales[np.argmax(agent.action_values)])
            
        
        eps*=0.998

        length = 0
        reward = None
        evidence = env.reset()
        agent.reset_channels()
        while env.iters < 200:
            e = np.random.uniform()
            choice = np.argmax(agent.action_values) if e >= eps else np.random.randint(0,agent.n_actions)
            
            agent.channels = agent.decay*agent.channels + agent.scales[choice]*evidence # The scale chosen is a measure of the agent's confidence in the decision
            prob = softmax(agent.channels)
            guess = np.argmax(prob) if np.max(prob) > 0.5 else None
            reward,evidence,done = env.step(guess)
            
            if done:
                agent.reset_channels()
                agent.action_values[choice] += alpha*(reward - agent.action_values[choice]) # Q-learning update on termination
            else:
                agent.action_values[choice] += alpha*(reward + np.max(agent.action_values) - agent.action_values[choice]) # Q-learning update before termination

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
            correct += int(reward != -30)
    corrects.append(correct/10.0)
    lengths.append(length/1000.0)
    rewards.append(avg_reward/1000.0)
    env.reset()
    print "eps = {} done in time {}. Final accuracy is {}%, average decision time is {} and average reward is {}".format(env.eps,time.time() - start_time,corrects[-1], lengths[-1], rewards[-1])
    
    return lengths,corrects,rewards,scales,q
   
def test(env_eps):
    env = ArgmaxEnv(env_eps)
    agent = Agent(env)
    corrects = []
    rewards = []
    lengths = []
    
    for i in range(10):   
        avg_reward = 0
        reward = None
        evidence = env.reset()
        agent.reset_channels()
        correct = 0
        length=0
        choice = i+1
        while env.iters < 5000:
            length+=1
            agent.channels = agent.decay*agent.channels + choice*evidence
            prob = softmax(agent.channels)
            guess = np.argmax(prob) if np.max(prob) > 0.5 else None
            reward,evidence,done = env.step(guess)
            avg_reward+=reward
            
            if done:
                agent.reset_channels()
                correct += int(reward != -30)
            
        corrects.append(correct/50.0)
        lengths.append(length/5000.0)
        rewards.append(avg_reward/5000.0)
        env.reset()
        
    return lengths,corrects,rewards

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='ArgMax Environment')
    parser.add_argument('--test', action='store_true',
                        help='Test')
    parser.add_argument('--eps', type=float, default=None, metavar='E',
                        help='Spead of the distribution')
    args = parser.parse_args()
    
    if not args.test:
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
            ret = p.map(train,zip(np.arange(10)/10.0,np.ones((10,))*0.001))
            
            lengths = [r[0] for r in ret]
            corrects = [r[1] for r in ret]
            rewards = [r[2] for r in ret]
            scales = [r[3] for r in ret]
            q = [softmax(np.array(r[-1]),axis=-1).T for r in ret]
            
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
            
            fig,axes = plt.subplots(len(q),sharex=True)
            for i,ax in enumerate(axes):
                ax.set_title("eps = %.2f" % (i/10.0))
                ax.imshow(q[i])
                
            fig,axes = plt.subplots(len(scales),sharex=True)
            for i,ax in enumerate(axes):
                ax.set_title("eps = %.2f" % (i/10.0))
                ax.plot(scales[i])
            
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