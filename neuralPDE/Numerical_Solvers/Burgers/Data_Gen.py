#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

---------------------------------------------------------------------------------------
Data Generation for 1D Burgers Surrogate
---------------------------------------------------------------------------------------

Code inspired from from Steve Brunton's FFT Example videos :
https://www.youtube.com/watch?v=hDeARtZdq-U&t=639s

Equation:
u_t + u*u_x =  nu*u_xx on [-1,1] x [-1,1]

Boundary Conditions: 
Periodic (implicit via FFT)

Initial Distribution :
u(x,y,t=0) = sin(alpha*pi*x) + np.cos(beta*np.pi*-x) + 1/np.cosh(gamma*np.pi*x) 
where alpha, beta, gamma all lie within the domain [-3,3]^3 sampled using a hypercube. 

Initial Velocity Condition :
u_t(x,y,t=0) = 0

"""

# %% 
#Importing the required packages
import os
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
from pyDOE import lhs 
from Burgers_1D import *
# %%
n_sims = 1000 #Total Number of simulation datapoints to be generated. 

#Grabbing the simulation parameters from the specified domain. 
 #alpha, beta, gamma
lb = np.asarray([-3, -3, -3]) # Lower Bound of the parameter domain
ub = np.asarray([3, 3, 3]) # Upper bound of the parameter domain

params = lb + (ub - lb) * lhs(3, n_sims)

Nx = 1000 #Number of x-points
Nt = 500 #Number of time instances 
x_min = 0.0 #Min of X-range 
x_max = 2.0 #Max of X-range 
t_end = 1.25 #Time Maximum
nu = 0.002

x_slice = 5
t_slice = 10
# %%
if __name__ == "__main__":
    sim = Burgers_1D(Nx, Nt, x_min, x_max, t_end, nu) 
    u_list = []
    for ii in tqdm(range(n_sims)):

        alpha = params[ii, 0]
        beta = params[ii, 1]
        gamma = params[ii, 2]

        sim.InitializeU(alpha, beta, gamma)
        u_sol, x, dt = sim.solve()
        u_list.append(u_sol)

    u_sol = np.asarray(u_list)[:, ::t_slice, ::x_slice]
    x = x[::x_slice]
    dt = dt*t_slice

    np.savez(os.getcwd() + "/Burgers_1d.npz", u=u_sol, x=x, dt=dt)

# %%
