#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 10:20:50 2018

@author: nownow
"""

import torch
from torch.optim import Optimizer

class EligibilityTraceSemiGradientTD(Optimizer):
    
    '''
    Implements eligibility traces for semigradient TD
    
    The traces are updated as:
        z = decay*z + grad(model)
    The parameters are updated as:
        p = p + lr*delta*z
        
        where delta is the TD error
    
    '''
    
    def __init__(self,params,lr=1e-2,decay=1):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Invalid decay rate: {}".format(decay))
        defaults = dict(lr=lr,decay=decay)
        super(EligibilityTrace, self).__init__(params, defaults)
        self.reset_traces()
        
#    def __setstate__(self,state):
#        super(EligibilityTrace,self).__setstate__(state)
    
    def step(self,delta=1):
        loss = None
        
        for i,group in enumerate(self.param_groups):
            lr = group['lr']
            decay = group['decay']
            
            for j,(p,z) in enumerate(zip(group['params'],group['trace'])):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                z = decay*z + d_p
                self.param_groups[i]['trace'][j] = z
                p.data.add_(lr*delta,z)
        return loss
    
    def reset_traces(self):
        for i,param_group in enumerate(self.param_groups):
            params = param_group['params']
            trace = [torch.zeros_like(p) for p in params]
            self.param_groups[i]['trace'] = trace