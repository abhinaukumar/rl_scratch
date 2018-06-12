# -*- coding: utf-8 -*-
"""
Created on Wed May 23 11:11:55 2018

@author: nownow
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
# Define a k armed Bandit. This is defined similar to Section 2.3

class Bandit():
    def __init__(self):
        self.k = 10
        self.q_star = np.random.normal(4,size=(self.k,))
        self.opt_action = np.argmax(self.q_star)
    def get_reward(self,arm):
        return np.random.normal(self.q_star[arm])

# Define a greedy sample-average learning technique

def greedy(bandit,steps=1000):
    R = []
    N = np.zeros(shape=(bandit.k,))
    Q = np.random.normal(size=(bandit.k,))
    A = []
    for step in range(steps):
        act = np.argmax(Q)
        A.append(act)
        N[act]+=1
        R.append(bandit.get_reward(act))
        Q[act] += (R[-1] - Q[act])/N[act]
    return (Q,R,A)

# Define an epsilon greedy sample-average learning technique

def e_greedy(bandit,eps=0.1,steps=1000,decay=0):
    
    R = []
    N = np.zeros(shape=(bandit.k,))
    Q = np.random.normal(size=(bandit.k,))
    A = []
    for step in range(steps):
        e = np.random.uniform()
        act = np.argmax(Q) if e > eps/(1+decay*step) else np.random.randint(0,bandit.k)
        A.append(act)
        N[act]+=1
        R.append(bandit.get_reward(act))
        Q[act] += (R[-1] - Q[act])/N[act]
    return (Q,R,A)

def run_greedy_experiments(bandits,steps,eps,decay):
    
    results = []
    
    if len(eps) != len(decay):
        print "Invalid arguments"
        return
    
    for i in range(len(eps)):
        
        print "eps =",eps[i],"decay =",decay[i]
        
        r_ret = []
        a_ret = []
        
        # Run sample averaging using greedy algorithm on bandits
        for j in range(len(bandits)):
            # Report progress every 100 iterations
            if not j%100:
                print j
            ret = e_greedy(bandits[j],eps[i],steps,decay[i])
            r_ret.append(ret[1])
            a_ret.append(ret[2])
            
        r_ret = np.array(r_ret)
        a_ret = np.array(a_ret)
        
        # Find average reward across bandits
        r_mean = np.mean(r_ret,axis=0)
        # Find average optimal action accuracy across bandits
        acc = np.array([np.sum(opt_actions == a_ret[:,k])*100.0/len(bandits) for k in range(steps)])
        results.append((r_mean.copy(),acc.copy()))
        
    return results
    
def plot_results(results,eps,decay):
    
    if len(results) != len(eps) or len(results) != len(decay):
        print "Invalid arguments"
        return

    # Plot average returns
    plt.figure()
    for i in range(len(results)):
        plt.semilogx(results[i][0],label='$\epsilon = ' + str(eps[i]) + ', decay = ' + str(decay[i]) + '$')
    plt.xlabel('Steps')
    plt.ylabel('Average returns')
    plt.legend()
    
    # Plot average optimal action accuracy
    plt.figure()
    for i in range(len(results)):
        plt.semilogx(results[i][1],label='$\epsilon = ' + str(eps[i]) + ', decay = ' + str(decay[i]) + '$')
    plt.xlabel('Steps')
    plt.ylabel('Average optimal action accuracy')
    plt.legend()
    
# Define a gradient bandit algorithm
    
def gradient_bandit(bandit,alpha=0.1,steps=1000,keep_baseline=True):
    
    keep_baseline = int(keep_baseline)
    R = []
    R_av = 0
    H = np.zeros((bandit.k,))
    pi = np.exp(H)/np.sum(np.exp(H))
    N = np.zeros(shape=(bandit.k,))
    Q = np.random.normal(size=(bandit.k,))
    A = []
    
    for step in range(steps):
        
        # Act based on Boltzmann distribution
        e = np.random.uniform()
        act = np.searchsorted(np.cumsum(pi),e)
        A.append(act)
        
        N[act]+=1
        R.append(bandit.get_reward(act))
        R_av= (R[-1]/(step+1) + R_av*step/(step+1))*keep_baseline
        
        Q[act] += (R[-1] - Q[act])/N[act]
        
        # Update action preferences
        H[act] += alpha*(R[-1] - R_av)*(1 - pi[act])
        H[:act] -= alpha*(R[-1] - R_av)*(pi[:act])
        H[act+1:] -= alpha*(R[-1] - R_av)*(pi[act+1:])
        
        pi = np.exp(H)/np.sum(np.exp(H))
        
    return (Q,R,A,H)

def run_gradient_bandit_experiments(bandits,steps,alpha,baseline):
    
    results = []
    
    if len(alpha) != len(baseline):
        print "Invalid arguments"
        return
        
    for i in range(len(alpha)):
        
        print "alpha =",alpha[i],"baseline =",baseline[i]
        
        r_ret = []
        a_ret = []
        
        # Run sample averaging using greedy algorithm on bandits
        for j in range(len(bandits)):
            # Report progress every 100 iterations
            if not j%100:
                print j
            ret = gradient_bandit(bandits[j],alpha[i],steps,baseline[i])
            r_ret.append(ret[1])
            a_ret.append(ret[2])
            
        r_ret = np.array(r_ret)
        a_ret = np.array(a_ret)
        
        # Find average reward across bandits
        r_mean = np.mean(r_ret,axis=0)
        # Find average optimal action accuracy across bandits
        acc = np.array([np.sum(opt_actions == a_ret[:,k])*100.0/len(bandits) for k in range(steps)])
        results.append((r_mean.copy(),acc.copy()))
        
    return results

def plot_gradient_bandit_results(results,alpha,baseline):
    
    if len(results) != len(alpha) or len(results) != len(baseline):
        print "Invalid arguments"
        return

    # Plot average returns
    plt.figure()
    for i in range(len(results)):
        plt.plot(results[i][0],label='alpha = ' + str(alpha[i]) + ', baseline = ' + str(baseline[i]))
    plt.xlabel('Steps')
    plt.ylabel('Average returns')
    plt.legend()
    
    # Plot average optimal action accuracy
    plt.figure()
    for i in range(len(results)):
        plt.plot(results[i][1],label='alpha = ' + str(alpha[i]) + ', baseline = ' + str(baseline[i]))
    plt.xlabel('Steps')
    plt.ylabel('Average optimal action accuracy')
    plt.legend()
    
# Initialize 1000 independent bandit problems
runs = 2000
steps = 1000
bandits = [Bandit() for i in range(runs)]
opt_actions = [bandit.opt_action for bandit in bandits]

alpha=[0.4,0.4,0.1,0.1]
baseline=[True,False,True,False]

print "Running experiments"

results = run_gradient_bandit_experiments(bandits,steps,alpha,baseline)

print "Plotting results"

plot_gradient_bandit_results(results,alpha,baseline)