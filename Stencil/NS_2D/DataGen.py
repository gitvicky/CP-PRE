#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Data Generation of the Navier-Stokes Turbulence 

"""

# %%
import os
import numpy as np
from pyDOE import lhs 
from tqdm import tqdm 
from time import time
from matplotlib import pyplot as plt 

from NS_Numerical import *


# %%
start_time = time()

n_sims = 100

lb = np.asarray([0.5, 0.5]) #a, b
ub = np.asarray([1.0, 1.0])

params = lb + (ub - lb) * lhs(2, n_sims*2) #Safety 20 

# %%
t_slice = 10 
x_slice = 4

#Running the simulation. 
ii=0
while ii < n_sims:
    solver= Navier_Stokes_2d(400, 0.0, 0.5, 0.001, 0.001, 1, params[ii, 0], params[ii, 1]) # N, t, tEnd, dt, nu, L, a , b  #(a and b are multiplier for the IC - ranging from 0.1 to 1, 1 being used by Philip)
    u, v, p, w, x, dt,err = solver.solve()
    if err == 0:
        np.savez(os.getcwd() + "/Data/NS_Spectral_" + str(ii) + ".npz", u=u[::t_slice, ::x_slice, ::x_slice], v=v[::t_slice, ::x_slice, ::x_slice], p=p[::t_slice, ::x_slice, ::x_slice], w=w[::t_slice, ::x_slice, ::x_slice], x=x[::x_slice], t=0.001*t_slice)
        ii+=1
        print(ii)

# %%
end_time = time()
print("Total Time : " + str(end_time - start_time))
# %%
#Cleaning up the runs into one. 
data_loc = os.getcwd() + '/Data/NS_Spectral_'

u_list = []
v_list = []
p_list = []
w_list = []

for ii in tqdm(range(n_sims)):
    u_list.append(np.load(data_loc + str(ii) + ".npz")['u'])
    v_list.append(np.load(data_loc + str(ii) + ".npz")['v'])
    p_list.append(np.load(data_loc + str(ii) + ".npz")['p'])
    w_list.append(np.load(data_loc + str(ii) + ".npz")['w'])

x = x[::x_slice]
dt = 0.001 * t_slice

# %%
u = np.asarray(u_list)
v = np.asarray(v_list)
p = np.asarray(p_list)
w = np.asarray(w_list)

# %% 
np.savez(os.getcwd() + "/Data/NS_Spectral_combined100.npz", u=u, v=v, p=p, w=w, x=x, dt=0.001*t_slice)

# %%
