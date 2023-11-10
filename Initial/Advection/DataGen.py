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
# %% 
'''
# 1D Advection Equation 
    U_t + v U_x = 0
where, v = [0.1, 1.0]

Exact Solver : 
  U(x,t) = exp(-200*(x-xc-v*t).^2)

  xc - location of the center
  v - velocity 

'''
from Advection_numerical import *
#Obtaining the exact and FD solution of the 1D Advection Equation. 
x_discretisation, t_end = 200, 0.5
n_sims = 100

lb = np.asarray([0.1, 0.1]) #pos, velocity
ub = np.asarray([1.0, 1.0])

params = lb + (ub - lb) * lhs(2, n_sims)

# %% 
u_sol = []
for ii in tqdm(range(n_sims)):

    solver = Advection_1d(xc=params[ii, 0], v=params[ii,1], N=x_discretisation, tmax=t_end)
    u_sol.append(solver.solve())

u_sol = np.asarray(u_sol)
u_sol = u_sol[:, :, 1:-2]
dt = solver.dt
dx = solver.dx
x = solver.x[1:-2]
t = np.linspace(0, solver.tmax, solver.nsteps)
v = params[:,1]
# %%

