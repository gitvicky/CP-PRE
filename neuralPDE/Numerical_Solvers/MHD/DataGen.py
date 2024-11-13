#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Data Generation of the Constrained MHD

"""

# %%
import os
import numpy as np
from pyDOE import lhs 
from tqdm import tqdm 
from time import time
from matplotlib import pyplot as plt 

from ConstrainedMHD_2D import *

N = 128
boxsize = 1.0
tEnd = 0.5

# %%
start_time = time()

n_sims = 500

lb = np.asarray([0.5, 0.5, 0.5]) #a, b, c
ub = np.asarray([1.0, 1.0, 1.0])

params = lb + (ub - lb) * lhs(3, n_sims*2) #Safety 20 

# %% 
import sys 
def update_status_bar(progress):
    bar_length = 20
    filled_length = int(bar_length * progress // 100)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f'\rProgress: [{bar}] {progress}%')
    sys.stdout.flush()
# %%
t_len = 5041
t_slice = 25
x_slice = 1

#Running the simulation. 
ii=0
while ii < n_sims:
    rho, u, v, p, Bx, By, dt, x, err = solve(N, boxsize, tEnd, params[ii, 0], params[ii, 1], params[ii, 2])
    rho, u, v, p, Bx, By, dt = rho[:t_len], u[:t_len], v[:t_len], p[:t_len], Bx[:t_len], By[:t_len], dt[:t_len]
    if err == 0:
        np.savez(os.getcwd() + "/Constrained_MHD_" + str(ii) + ".npz", rho=rho[::t_slice, ::x_slice, ::x_slice], u=u[::t_slice, ::x_slice, ::x_slice], v=v[::t_slice, ::x_slice, ::x_slice], p=p[::t_slice, ::x_slice, ::x_slice], Bx=Bx[::t_slice, ::x_slice, ::x_slice], By=By[::t_slice, ::x_slice, ::x_slice], x=x[::x_slice], dt=dt[0]*t_slice)
        ii+=1
        progress = int(ii / n_sims * 100)
        update_status_bar(progress)

# %%
end_time = time()
print("Total Time : " + str(end_time - start_time))
# %%
#Cleaning up the runs into one. 
data_loc = os.getcwd() + '/Constrained_MHD_'

rho_list = []
u_list = []
v_list = []
p_list = []
Bx_list = []
By_list = []

for ii in tqdm(range(n_sims)):
    rho_list.append(np.load(data_loc + str(ii) + ".npz")['u'])
    u_list.append(np.load(data_loc + str(ii) + ".npz")['u'])
    v_list.append(np.load(data_loc + str(ii) + ".npz")['v'])
    p_list.append(np.load(data_loc + str(ii) + ".npz")['p'])
    Bx_list.append(np.load(data_loc + str(ii) + ".npz")['Bx'])
    By_list.append(np.load(data_loc + str(ii) + ".npz")['By'])
    try: 
        os.remove(data_loc + str(ii) + ".npz")
    except:
        pass

# %%
rho = np.asarray(rho_list)
u = np.asarray(u_list)
v = np.asarray(v_list)
p = np.asarray(p_list)
Bx = np.asarray(Bx_list)
By = np.asarray(By_list)
# %% 
dx = boxsize / N
vol = dx**2
xlin = np.linspace(0.5*dx, boxsize-0.5*dx, N)
x = xlin

np.savez(os.getcwd() + "/Constrained_MHD_combined.npz", rho=rho, u=u, v=v, p=p, Bx=Bx, By=By, x=x, dt=dt)

# %%
