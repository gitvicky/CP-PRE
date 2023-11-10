#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 15:04:51 2022

@author: vgopakum

u_t = D.u_xx +u.D_x - c.u_x
"""

# %% 

import os 
import numpy as np 
from matplotlib import pyplot as plt
from time import time
from sympy import *


class Conv_Diff_1d:
    
    def __init__(self, dx, dt, D_damp, c, mu, sigma):
        
        self.dx = 0.05
        self.x = np.arange(0, 10, self.dx)
        self.x_length = len(self.x)
        self.dt = 0.0005

        self.D_damp = D_damp
        self.c = c 
        self.mu = mu
        self.sigma = sigma 

        self.initializeU()
        self.initializeD()
        self.initializeParams()

        
        self.u_sol = []
        self.u_exact = []
        
    def initializeU(self):
        self.u_0 = np.exp(-(self.x-self.mu)**2 / (2*self.sigma**2)) / np.sqrt(2*np.pi*self.sigma**2)
        
        
    def initializeD(self):
        x_sym = Symbol('x')
        D_sym = sin(x_sym/self.D_damp)
        D_func = lambdify(x_sym, D_sym, "numpy") 
        self.D = D_func(self.x)
        dD_dx_sym = diff(D_sym, x_sym)
        dD_dx_func = lambdify(x_sym, dD_dx_sym, "numpy") 
        self.dD_dx = dD_dx_func(self.x)

        # #Fixing the Diffusion Coefficient 
        self.D = np.ones(200) * 1.0
        self.dD_dx = np.zeros(200)
            
    def initializeParams(self):        
        self.alpha_diff = (self.D*self.dt)/self.dx**2
        self.alpha_conv = self.c*self.dt/(2*self.dx)
        

#Implementation of FTCS scheme to build the test dataset No flux boundary conditions
    def solve(self):
        n_itim = 5000
        u_dataset = np.zeros((n_itim+1, self.x_length))
        u_dataset[0] = self.u_0
        u = self.u_0.copy()    

        t = np.arange(0, n_itim*self.dt +self.dt, self.dt)
        t_max = np.amax(t)
        t_min = np.amin(t)
        t_norm = (t - t_min) / (t_max - t_min)

        u = self.u_0

        # plot = plt.figure()
        for ii in range(n_itim):
            u[1:-1] = (u[1:-1]*(1 - 2*self.alpha_diff[1:-1]) 
                    + u[2:]*(self.alpha_diff[2:] + self.dD_dx[2:]*(self.dt/(2*self.dx)) - self.alpha_conv)
                    + u[:-2]*(self.alpha_diff[:-2] - self.dD_dx[:-2]*(self.dt/(2*self.dx)) + self.alpha_conv)
                    )    
            u[0] = u[1]
            u[-1] = u[-2]
            
            u_dataset[ii+1] = u

        return u_dataset, self.D, self.dD_dx
    

if __name__ == "__main__":
    dx = 0.05
    dt = 0.0005
    D_damp = 2*np.pi
    c = 0.5
    mu = 5
    sigma = 0.5 

    sim = Conv_Diff_1d(dx, dt, D_damp, c, mu, sigma) 
    u_sol, D, D_x = sim.solve()
# %%
