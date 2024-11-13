#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D wave equation via FFT 

u_tt = c^2 * (u_xx + u_yy)

on [-1, 1]x[-1, 1], t > 0 and Dirichlet BC u=0

Based on: http://people.bu.edu/andasari/courses/numericalpython/python.html
"""
# %%
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm  
from tqdm import tqdm 

from Wave_2D_Spectral import Wave_2D


#Example of Usage
Nx = 64 # Mesh Discretesiation 
x_min = -1.0 # Minimum value of x
x_max = 1.0 # maximum value of x
y_min = -1.0 # Minimum value of y 
y_max = 1.0 # Minimum value of y
tend = 1
# Lambda = 20
# aa = 0.25
# bb = 0.25
c = 1.0 # Wave Speed <=1.0


#Initialising the Solver
solver = Wave_2D(Nx, x_min, x_max, tend, c)

# %%       
def Parameter_Scan ():
    #Simulation Data built via parameter scans
    Lambda_range = np.arange(10,51,5)
    a_range = np.arange(0.1, 0.51, 0.1)
    b_range = np.arange(0.1, 0.51, 0.1)
    
    #IC = np.exp(-Lambda*((xx-a)**2 + (yy-b)**2))
    
    list_u = []
    list_ic = []
    
    for ii in tqdm(Lambda_range):
        for jj in a_range:
            for kk in b_range:

                x, y, t, u = solver.solve(ii, jj, kk)
                
                list_ic.append([ii, jj, kk])
                list_u.append(u)
        
    ic = np.asarray(list_ic)
    u = np.asarray(list_u)
    
    # np.savez('Spectral_Wave_data_Parameter_Scan.npz', x=x, y=y,t=t, u=u, ic=ic)
    
    # data = np.load('Spectral_Wave_data.npz')
    # file_names = data.files

# %%
def LHS_Sampling(N=10):
    #Simulation Data Built using LHS sampling
    from pyDOE import lhs
    
    lb = np.asarray([10, 0.10, 0.10]) #Lambda, a, b 
    ub = np.asarray([50, 0.50, 0.50]) #Lambda, a, b 
    
       
    param_lhs = lb + (ub-lb)*lhs(3, N)
    
    list_u = []
    
    for ii in tqdm(range(N)):
                x, y, t, u = solver.solve(param_lhs[ii, 0], param_lhs[ii, 1], param_lhs[ii, 2])
                t = t[::5]
                list_u.append(u[::5])
        
    ic = param_lhs
    u = np.asarray(list_u)
    
    np.savez('Spectral_Wave_data_LHS.npz', x=x, y=y,t=t, u=u, ic=ic)

# %%
LHS_Sampling(1000)