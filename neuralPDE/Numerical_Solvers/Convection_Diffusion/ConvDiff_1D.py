#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
23 Aug 2022

@author: vgopakum

This code solves the 1D convection diffusion equation 

u_t = D.u_xx +u.D_x - c.u_x

with spatially varying diffusion coefficient (sin(x/damping_factor)).
It uses a finite difference solver with no flux boundary conditions. 

The code is currently setup to model the evolution of a 1D gaussian. 

"""
# %% 

import os 
import numpy as np 
from matplotlib import pyplot as plt
from time import time
from sympy import *


class Conv_Diff_1d:
    
    def __init__(self, Nx, Nt, x_min, x_max, t_end, D_damp, c, mu, sigma):
        """
    Initialize the Conv_Diff_1d class.

    Args:
        Nx (int): Numerical discretiation in space. 
        Nt (int): Numerical discretiation in time. 
        x_min (float): The minimum value of the spatial domain.
        x_max (float): The maximum value of the spatial domain.
        t_end (float): The maximum value of the time domain.
        D_damp (float): The damping coefficient for the diffusion term (sin(x)/D_damp).
        c (float): The velocity of the convection term.
        mu (float): The mean value for the initial condition.
        sigma (float): The standard deviation for the initial condition.
    """       
        self.x_length = Nx
        self.dx = (x_max - x_min)/self.x_length
        self.x = np.arange(x_min, x_max, self.dx)

        self.t_length = Nt
        self.dt = (t_end)/self.t_length
        self.t = np.arange(0, t_end, self.dt)
        self.n_itim = Nt

        self.D_damp = D_damp
        self.c = c 
        self.mu = mu
        self.sigma = sigma 

        self.initializeU()#Initialising the Initial Conditions
        self.initializeD()#Setting up the spatially varying Diffusion Coefficient. 
        self.initializeParams()#Setting up the CFL states for FD
        
        self.u_sol = []
        self.u_exact = []
        
    def initializeU(self):
        #Initialising the initial field using a 1D gaussian with mu for mean and sigma for variance. 
        #
        self.u_0 = np.exp(-(self.x-self.mu)**2 / (2*self.sigma**2)) / np.sqrt(2*np.pi*self.sigma**2)
        
        
    def initializeD(self):
        #Initialising the spatially varying diffusion coefficient 
        #whose derivatives are estimated using sympy

        x_sym = Symbol('x')
        D_sym = sin(x_sym/self.D_damp)
        D_func = lambdify(x_sym, D_sym, "numpy") 
        self.D = D_func(self.x)
        dD_dx_sym = diff(D_sym, x_sym)
        dD_dx_func = lambdify(x_sym, dD_dx_sym, "numpy") 
        self.dD_dx = dD_dx_func(self.x)

        # #Fixing the Diffusion Coefficient 
        self.D = np.ones(self.x_length) * 1.0
        self.dD_dx = np.zeros(self.x_length)
            
    def initializeParams(self): 
        #Intialising the convection and diffusion parameters        
        self.alpha_diff = (self.D*self.dt)/self.dx**2
        self.alpha_conv = self.c*self.dt/(2*self.dx)
        
        # Check the CFL condition
        cfl_diff = np.max(self.alpha_diff)
        cfl_conv = np.max(self.alpha_conv)
        cfl = cfl_diff + cfl_conv
        print(f"CFL number: {cfl}")
        assert cfl <= 1, "CFL condition violated" 

#Implementation of FTCS scheme to build the test dataset No flux boundary conditions
    def solve(self):
        u_dataset = np.zeros((self.n_itim+1, self.x_length))
        u_dataset[0] = self.u_0
        u = self.u_0.copy()    

        t = np.arange(0, self.n_itim*self.dt +self.dt, self.dt)
        t_max = np.amax(t)
        t_min = np.amin(t)
        t_norm = (t - t_min) / (t_max - t_min)

        u = self.u_0

        # plot = plt.figure()
        for ii in range(self.n_itim):
            u[1:-1] = (u[1:-1]*(1 - 2*self.alpha_diff[1:-1]) 
                    + u[2:]*(self.alpha_diff[2:] + self.dD_dx[2:]*(self.dt/(2*self.dx)) - self.alpha_conv)
                    + u[:-2]*(self.alpha_diff[:-2] - self.dD_dx[:-2]*(self.dt/(2*self.dx)) + self.alpha_conv)
                    )    
            u[0] = u[1]
            u[-1] = u[-2]
            
            u_dataset[ii+1] = u

        return u_dataset, self.D, self.dD_dx, self.x, self.dt
    

if __name__ == "__main__":
    #Example Usage
    Nx = 256 #Number of x-points
    Nt = 5000 #Number of time instances 
    x_min = 0.0 #Min of X-range 
    x_max = 10.0 #Max of X-range 
    t_end = 2.5 #Time Maximum
    D_damp = 2*np.pi #Damping Factor
    c = 0.5 #Convection velocity 
    mu = 5 #Gaussian mean
    sigma = 0.5 #Gaussian Variance

    sim = Conv_Diff_1d(Nx, Nt, x_min, x_max, t_end, D_damp, c, mu, sigma) 
    u_sol, D, D_x, x, dt = sim.solve()
# %%
