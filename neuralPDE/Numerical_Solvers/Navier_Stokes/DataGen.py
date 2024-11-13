#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Data Generation of the Navier-Stokes Turbulence using a Finite Volume Solver - Kelvin Helmholtz Instability
"""

# %%
import os
import numpy as np
from pyDOE import lhs 
from tqdm import tqdm 
from time import time
from matplotlib import pyplot as plt 

from NS_2D_spectral import *
from NS_2D_FV import * 
 
solver = 'FV' #Finite Volume Scheme 
# solver = 'Spectral' #Spectral Scheme 
# %%
start_time = time()

n_sims = 500

if solver == 'Spectral':
    lb = np.asarray([0.5, 0.5]) #a, b
    ub = np.asarray([1.0, 1.0])
elif solver == 'FV':
    lb = np.asarray([0.1, 1.0]) #Vy_ic_params
    ub = np.asarray([1.0, 10.0]) #P_ic
    
params = lb + (ub - lb) * lhs(2, n_sims*2) #Safety 20 

# %%

if solver == 'Spectral':
    t_slice = 10 
    x_slice = 4

    #Running the simulation. 
    ii=0
    while ii < n_sims:
        solver= Navier_Stokes_2d(400, 0.0, 0.5, 0.001, 0.001, 1, params[ii, 0], params[ii, 1]) # N, t, tEnd, dt, nu, L, a , b  #(a and b are multiplier for the IC - ranging from 0.1 to 1, 1 being used by Philip)
        u, v, p, w, x, dt,err = solver.solve()
        if err == 0:
            np.savez(os.getcwd() + "/NS_Spectral_" + str(ii) + ".npz", u=u[::t_slice, ::x_slice, ::x_slice], v=v[::t_slice, ::x_slice, ::x_slice], p=p[::t_slice, ::x_slice, ::x_slice], w=w[::t_slice, ::x_slice, ::x_slice], x=x[::x_slice], t=0.001*t_slice)
            ii+=1


    end_time = time()
    print("Total Time : " + str(end_time - start_time))

    #Cleaning up the runs into one. 
    data_loc = os.getcwd() + '/NS_Spectral_'

    u_list = []
    v_list = []
    p_list = []
    w_list = []

    for ii in tqdm(range(n_sims)):
        u_list.append(np.load(data_loc + str(ii) + ".npz")['u'])
        v_list.append(np.load(data_loc + str(ii) + ".npz")['v'])
        p_list.append(np.load(data_loc + str(ii) + ".npz")['p'])
        w_list.append(np.load(data_loc + str(ii) + ".npz")['w'])
        try: 
            os.remove(data_loc + str(ii) + ".npz")
        except:
            pass
        
    x = x[::x_slice]
    dt = 0.001 * t_slice

    u = np.asarray(u_list)
    v = np.asarray(v_list)
    p = np.asarray(p_list)
    w = np.asarray(w_list)

    np.savez(os.getcwd() + "/NS_Spectral_combined.npz", u=u, v=v, p=p, w=w, x=x, dt=0.001*t_slice)

# %% 

if solver == 'FV':

        # Simulation parameters
    N                      = 128 # resolution
    boxsize                = 1.
    tStart                 = 0
    tEnd                   = 2.0
    gamma                  = 5/3 # ideal gas gamma
    courant_fac            = 0.4
    # vy_ic                  = 0.1 # initial vy parameterisations
    # p_ic                   = 2.5 # initial pressure value 

    t_slice = 40    
    x_slice = 1

    #Running the simulation. 
    ii=0
    while ii < n_sims:
        rho, uu, vv, pp, dx, dtt = KelvinHelmholtz(N, boxsize, tStart, tEnd, gamma, courant_fac, params[ii, 0], params[ii, 1])
        np.savez(os.getcwd() + "/NS_FV_" + str(ii) + ".npz", u=uu[::t_slice, ::x_slice, ::x_slice], v=vv[::t_slice, ::x_slice, ::x_slice], p=pp[::t_slice, ::x_slice, ::x_slice], rho=rho[::t_slice, ::x_slice, ::x_slice], dx=dx*x_slice, dtt=dtt[0]*t_slice)
        ii+=1

    end_time = time()
    print("Total Time : " + str(end_time - start_time))

    #Cleaning up the runs into one. 
    data_loc = os.getcwd() + '/NS_FV_'

    u_list = []
    v_list = []
    p_list = []
    rho_list = []

    for ii in tqdm(range(n_sims)):
        u_list.append(np.load(data_loc + str(ii) + ".npz")['u'])
        v_list.append(np.load(data_loc + str(ii) + ".npz")['v'])
        p_list.append(np.load(data_loc + str(ii) + ".npz")['p'])
        rho_list.append(np.load(data_loc + str(ii) + ".npz")['rho'])
        try: 
            os.remove(data_loc + str(ii) + ".npz")
        except:
            pass
        
    dx = dx * x_slice
    dt = dtt[0] * t_slice

    u = np.asarray(u_list)
    v = np.asarray(v_list)
    p = np.asarray(p_list)
    rho = np.asarray(rho_list)

    np.savez(os.getcwd() + "/NS_FV_combined.npz", u=u, v=v, p=p, rho=rho, dx=dx, dt=dt)

    # %% 