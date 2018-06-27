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
    return np.exp(x)/np.expand_dims(np.sum(np.exp(x),axis=axis),axis=-1)

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
            reward = 30.0 - (self.prediction_time - 1) if guess == self.true_option else -30.0
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
        self.n_scales = 10
        self.n_thresholds = 5
        self.n_actions = self.n_scales * self.n_thresholds
        self.channels = np.zeros((self.n_options,))
        self.decay = 1.0
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
    eps = 1.0
    q = []
    scales = []
        
    for i in range(500):
        
        if not i%5:
            
            # Report the performance of the greedy policy every 50 iterations
            
            tot_reward = 0
            reward = None
            evidence = env.reset()
            agent.reset_channels()
            correct = 0
            length = 0
            choice = np.argmax(agent.action_values)
            
            scale = 1 + choice/agent.n_thresholds
            threshold = (10 - agent.n_thresholds + choice%agent.n_thresholds)/10.0
            
            _lengths = []
            _rewards = []
            
            while env.iters < 1000:
                length+=1
                agent.channels = agent.decay*agent.channels + scale*evidence
                prob = softmax(agent.channels)
                guess = np.argmax(prob) if np.max(prob) > threshold else None
                reward,evidence,done = env.step(guess)
                tot_reward+=reward
                
                if done:
                    _lengths.append(length)
                    length = 0
                    _rewards.append(reward)
                    agent.reset_channels()
                    correct += int(reward != -30)
                    
            if not i%50:
                print "Iteration {} in environment with eps {}".format(i,env.eps)
                
            corrects.append(correct/10.0)
            lengths.append([np.mean(_lengths),np.mean(_lengths) - np.std(_lengths),np.mean(_lengths) + np.std(_lengths)])
            rewards.append([np.mean(_rewards),np.mean(_rewards) - np.std(_rewards),np.mean(_rewards) + np.std(_rewards)])
        
            q.append(agent.action_values.copy())
            scales.append(1 + np.argmax(agent.action_values))
            
        
        eps*=0.95

        length = 0
        reward = None
        evidence = env.reset()
        agent.reset_channels()
        while env.iters < 500:
            e = np.random.uniform()
            choice = np.argmax(agent.action_values) if e >= eps else np.random.randint(0,agent.n_actions)
            scale = 1 + choice/agent.n_thresholds
            threshold = (10 - agent.n_thresholds + choice%agent.n_thresholds)/10.0
            agent.channels = agent.decay*agent.channels + scale*evidence # The scale chosen is a measure of the agent's confidence in the decision
            prob = softmax(agent.channels)
            guess = np.argmax(prob) if np.max(prob) > threshold else None
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
    scale = 1 + choice/agent.n_thresholds
    threshold = (10 - agent.n_thresholds + choice%agent.n_thresholds)/10.0
    _lengths = []
    _rewards = []
    
    while env.iters < 1000:
        length+=1
        agent.channels = agent.decay*agent.channels + scale*evidence
        prob = softmax(agent.channels)
        guess = np.argmax(prob) if np.max(prob) > threshold else None
        reward,evidence,done = env.step(guess)
        avg_reward+=reward
        
        if done:
            agent.reset_channels()
            _lengths.append(length)
            length = 0
            _rewards.append(reward)
            correct += int(reward != -30)
    corrects.append(correct/10.0)
    lengths.append([np.mean(_lengths),np.mean(_lengths) - np.std(_lengths),np.mean(_lengths) + np.std(_lengths)])
    rewards.append([np.mean(_rewards),np.mean(_rewards) - np.std(_rewards),np.mean(_rewards) + np.std(_rewards)])
    env.reset()
    print "eps = {} done in time {}. Final accuracy is {}%, average decision time is {} and average reward is {}".format(env.eps,time.time() - start_time,corrects[-1], lengths[-1], rewards[-1])
    
    return lengths,corrects,rewards,scales,q,agent

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='ArgMax Environment')
    parser.add_argument('--test', action='store_true',
                        help='Test')
    parser.add_argument('--eps', type=float, default=None, metavar='E',
                        help='Spead of the distribution')
    args = parser.parse_args()
    
    if args.eps != None:
        ret = train((args.eps,1e-4))
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
        
    else:
        p = Pool(10)
        ret = p.map(train,zip(np.arange(10)/10.0,np.ones((10,))*2e-2))
        lengths = [np.array(r[0])[:,0] for r in ret]
        l_min = [np.array(r[0])[:,1] for r in ret]
        l_max = [np.array(r[0])[:,2] for r in ret]
        corrects = [r[1] for r in ret]
        rewards = [np.array(r[2])[:,0] for r in ret]
        r_min = [np.array(r[2])[:,1] for r in ret]
        r_max = [np.array(r[2])[:,2] for r in ret]
        scales = [r[3] for r in ret]
        
        plt.figure()
        plt.title("Accuracy vs Time")
        for i in range(len(corrects)):
            plt.plot(corrects[i],label="eps = %.2f" % (i/10.0))
        plt.legend()
        
        plt.figure()
        plt.title("Delay vs Time")
        for i in range(len(lengths)):
            plt.plot(lengths[i],label="eps = %.2f" % (i/10.0))
            #plt.fill_between(np.arange(len(lengths[i])), l_min[i], l_max[i], alpha=0.5)
        plt.legend()
        
        plt.figure()
        plt.title("Reward vs Time")
        for i in range(len(rewards)):
            plt.plot(rewards[i],label="eps = %.2f" % (i/10.0))
            #plt.fill_between(np.arange(len(rewards[i])), r_min[i], r_max[i], alpha=0.5)
        plt.legend()
        
# After modifying environment
#
#    Optimal choice for environment with epsilon 0.0 is scale 3 and threshold 0.5 with average reward 30.0, accuracy 100.0 and delay 1.0
#    Optimal choice for environment with epsilon 0.1 is scale 3 and threshold 0.9 with average reward 28.6345, accuracy 99.73 and delay 2.208
#    Optimal choice for environment with epsilon 0.2 is scale 2 and threshold 0.8 with average reward 27.883, accuracy 99.55 and delay 2.8536
#    Optimal choice for environment with epsilon 0.3 is scale 2 and threshold 0.8 with average reward 26.8661, accuracy 98.67 and delay 3.3646
#    Optimal choice for environment with epsilon 0.4 is scale 2 and threshold 0.9 with average reward 25.7843, accuracy 99.45 and delay 4.9076
#    Optimal choice for environment with epsilon 0.5 is scale 2 and threshold 0.9 with average reward 24.1836, accuracy 98.44 and delay 5.9608
#    Optimal choice for environment with epsilon 0.6 is scale 2 and threshold 0.9 with average reward 21.4082, accuracy 95.93 and delay 7.4228
#    Optimal choice for environment with epsilon 0.7 is scale 2 and threshold 0.9 with average reward 16.2763, accuracy 90.31 and delay 9.9324
#    Optimal choice for environment with epsilon 0.8 is scale 1 and threshold 0.6 with average reward 6.7809, accuracy 76.94 and delay 14.0211
#    Optimal choice for environment with epsilon 0.9 is scale 1 and threshold 0.5 with average reward -9.8164, accuracy 42.22 and delay 14.2716