#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

---------------------------------------------------------------------------------------
Generating 1D Advection Dataset for PDE Surrogate Modelling 
---------------------------------------------------------------------------------------

"""
# %% 
#Importing the necessary packages 
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
from pyDOE import lhs 

from Advection_1D import * 
# %%
#Obtaining the exact and FD solution of the 1D Advection Equation. 

Nx = 100 #Number of x-points
Nt = 50 #Number of time instances 
x_min, x_max = 0.0, 2.0 #X min and max
t_end = 0.5 #time length
# v = 1 #Advection velocity 
# xc = 0.25 #Centre of Gaussian 

sim = Advection_1d(Nx, Nt, x_min, x_max, t_end) 

n_sims = 1000

lb = np.asarray([0.1, 50]) #pos, amplitude
ub = np.asarray([1.0, 200])

params = lb + (ub - lb) * lhs(2, n_sims)

# %% 
u_sol = []
for ii in tqdm(range(n_sims)):
    xc = params[ii, 0]
    amp = params[ii, 1]
    x, t, u_soln, u_exact = sim.solve(xc, amp, v=1)

    u_sol.append(u_soln)

u_sol = np.asarray(u_sol)
u_sol = u_sol[:, :, 1:-2]
velocity = params[:,1]

# %%

