#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Data Generation of the Convection-Diffusion PDE Solutions
u_t = D.u_xx +u.D_x - c.u_x

"""

# %%
import os
import numpy as np
from pyDOE import lhs 
from tqdm import tqdm 
from time import time
from matplotlib import pyplot as plt 
from ConvDiff_1D import * 

# %%
start_time = time()

n_sims = 1000

#Two different parameterisations are explored. 
lb = np.asarray([np.pi, 0.1, 1.0, 0.25]) #D, c, mu, sigma
ub = np.asarray([2*np.pi, 0.5, 8.0, 0.75])

# lb = np.asarray([2*np.pi, 0.5, 1.0, 0.25]) #D, c, mu, sigma
# ub = np.asarray([4*np.pi, 1.0, 8.0, 0.75])

params = lb + (ub - lb) * lhs(4, n_sims)

# %%
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

t_slice = 50

u_dataset = []  
D_dataset = []
D_x_dataset = []
for ii in tqdm(range(n_sims)):
    sim = Conv_Diff_1d(Nx, Nt, x_min, x_max, t_end, params[ii,0], params[ii,1], params[ii,2], params[ii,3])
    u_sol, D, D_x, x, dt = sim.solve()    
    u_dataset.append(u_sol)
    D_dataset.append(D)
    D_x_dataset.append(D_x)

uu = np.asarray(u_dataset)
D = np.asarray(D_dataset)
D_x = np.asarray(D_x_dataset)
# %%
np.savez(os.getcwd() + '/ConvDiff_u_1.npz', u = uu, D=D, D_x = D_x, x=x, dt=dt, params=params)

end_time = time()
print("Total Time : " + str(end_time - start_time))
# %%
