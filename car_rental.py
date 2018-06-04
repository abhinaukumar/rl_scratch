# -*- coding: utf-8 -*-
"""
Created on Fri May 25 09:48:52 2018

@author: nownow
"""

import numpy as np
import scipy.stats

class CarRental():
    def __init__(self):
        self.cars = np.array([20,20])
    
    # Number of rental requests and returned cars both follow Poisson distributions at both places
    
    def borrow_cars(self):
        return np.array([np.random.poisson(3),np.random.poisson(4)])
    def return_cars(place):
        return np.array([np.random.poisson(3),np.random.poisson(2)])
        
    def move_cars(self,cars_moved):
        cars_moved = np.clip(cars_moved,-5,5)
        self.cars += np.array([cars_moved,-cars_moved])
        np.clip(self.cars,0,20)
        return 2*cars_moved
    
    def run_business(self):
        borrowed = self.borrow_cars()
        returned = self.return_cars()
        
        if (borrowed>self.cars).any():
            return -np.inf
        
        self.cars += returned - borrowed
        return 10*np.sum(borrowed)

def policy_iter(rental):
    pi = np.zeros((21,21))
    V = np.zeros((21,21))
    actions = np.arange(-5,6)
    
    p_borrow_0 = scipy.stats.poisson.pmf(np.arange(0,11),3)
    p_borrow_1 = scipy.stats.poisson.pmf(np.arange(0,11),4)
    p_return_0 = p_borrow_0.copy()
    p_return_1 = scipy.stats.poisson.pmf(np.arange(0,11),2)
    
    np.unravel_index(np.argmax(a, axis=None), a.shape)
    delta = 5
    while(delta>1e-2):
        old_values = V.copy()
        for state,value in np.ndenumerate(V):
            V[state] =  