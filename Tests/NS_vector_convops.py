#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmarking out the Vector ConvOps over the regular ConvOps utilities across the NS Equations  
"""

# %%
# configuration = {"Case": 'Navier-Stokes',
#                  "Field": 'u, v, p, w',
#                  "Model": 'FNO',
#                  "Epochs": 500,
#                  "Batch Size": 5,
#                  "Optimizer": 'Adam',
#                  "Learning Rate": 0.005,
#                  "Scheduler Step": 100,
#                  "Scheduler Gamma": 0.5,
#                  "Activation": 'GeLU',
#                  "Physics Normalisation": 'No',
#                  "Normalisation Strategy": 'Min-Max',
#                  "T_in": 1,    
#                  "T_out": 20,
#                  "Step": 1,
#                  "Width_time": 16, 
#                  "Width_vars": 0,  
#                  "Modes": 8,
#                  "Variables":4, 
#                  "Loss Function": 'LP',
#                  "UQ": 'None', #None, Dropout
#                  "n_cal": 100, 
#                  "n_pred": 100
#                  }
# %% 
#Importing the necessary packages
import os 
import sys
import numpy as np
from tqdm import tqdm 
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import time 
from timeit import default_timer
from tqdm import tqdm 

# %%
#Importing the models and utilities. 
import sys
sys.path.append("..")
from Neural_PDE.Models.FNO import *
from Neural_PDE.Utils.processing_utils import * 
from Neural_PDE.Utils.training_utils import * 
from Neural_PDE.UQ.inductive_cp import * 
from Utils.plot_tools import subplots_2d

#Setting up locations. 
file_loc = os.getcwd()
data_loc = os.path.dirname(os.getcwd()) + '/Neural_PDE/Data'
model_loc = file_loc + '/Weights'
plot_loc = file_loc + '/Plots'
#Setting up the seeds and devices
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
# %% 

# Utility Functions 
#Stacking the various fields 
def stacked_fields(variables):
    stack = []
    for var in variables:
        var = torch.tensor(var, dtype=torch.float32) #Converting to Torch
        var = var.permute(0, 2, 3, 1) #Permuting to be BS, Nx, Ny, Nt
        stack.append(var)
    stack = torch.stack(stack, dim=1)
    return stack


def unstack_fields(field, axis, variable_names):
    fields = torch.split(field, 1, dim=axis)
    fields = [t.squeeze(axis) for t in fields]
    
    if len(fields) != len(variable_names):
        raise ValueError("Number of tensors and variable names should match.")
    
    variables = []
    for field in fields:
        variables.append(field.permute(0, 3, 1, 2))
    
    return variables

#Loading the Data
t1 = default_timer()
data =  np.load(data_loc + '/NS_Spectral_combined.npz')

u = data['u'].astype(np.float32)
v = data['v'].astype(np.float32)
p = data['p'].astype(np.float32)
w = data['w'].astype(np.float32)
x = data['x'].astype(np.float32)
y = data['x'].astype(np.float32)
dt = data['dt'].astype(np.float32)
nu = 0.001#kinematic viscosity

vars = stacked_fields([u,v,p])

# %%
#Define Bounds
lb = np.asarray([0.5, 0.5]) #Vx - aa, Vy - bb
ub = np.asarray([1.0, 1.0])

dx = np.asarray(x[-1] - x[-2])
dy = dx
dt = dt

alpha = 1/dt*2
beta = 1/dx*2
gamma = 1/dx**2                 

alpha, beta, gamma = 1,1,1

from Utils.ConvOps_2d import ConvOperator
#Defining the required Convolutional Operations. 
D_t = ConvOperator(domain='t', order=1)#, scale=alpha)
D_x = ConvOperator(domain='x', order=1)#, scale=beta) 
D_y = ConvOperator(domain='y', order=1)#, scale=beta)
D_x_y = ConvOperator(domain=('x', 'y'), order=1)#, scale=beta)
D_xx_yy = ConvOperator(domain=('x','y'), order=2)#, scale=gamma)

#Continuity
def residual_continuity(vars, boundary=False):
    u, v = vars[:,0], vars[:, 1]
    res = D_x(u) + D_y(v)
    if boundary:
        return res
    else: 
        return res[...,1:-1,1:-1,1:-1]
    
#Momentum 
def residual_momentum(vars, boundary=False):
    u, v, p = vars[:,0], vars[:, 1], vars[:, 2]

    res_x = D_t(u) + u*D_x(u) + v*D_y(u) - nu*D_xx_yy(u) + D_x(p)
    res_y = D_t(v) + u*D_x(v) + v*D_y(v) - nu*D_xx_yy(v) + D_y(p)

    if boundary:
        return res_x + res_y
    else: 
        return res_x[...,1:-1,1:-1,1:-1] + res_y[...,1:-1,1:-1,1:-1]

# %% 
#Using vector operations 
from Utils.VectorConvOps import *
div = Divergence(scale=1) 

def residual_continuity_vector(vars, boundary=False):
    u, v = vars[:,0], vars[:, 1]
    res = div(u, v)
    if boundary:
        return res
    else: 
        return res[...,1:-1,1:-1,1:-1]
    
grad = Gradient()
laplace = Laplace()
grad_time = Gradient(domain=('t', 't'))

def residual_momentum_vector(vars, boundary=False):
    u, v, p = vars[:,0], vars[:, 1], vars[:, 2]
    res = grad_time(u,v) + vectorize(dot(vectorize(u,v), grad(u)), dot(vectorize(u,v), grad(v))) - nu*laplace(u,v) + grad(p)
    res = torch.sum(res, dim=0)
    if boundary:
        return res
    else: 
        return res[...,1:-1,1:-1,1:-1]
    

# %% #Checking continuity 
res_cont = residual_continuity(vars)
res_cont_vect = residual_continuity_vector(vars)

#Plotting the error bars. 
idx = 5
t_idx = 10
values = [
          res_cont[idx, ...,t_idx],
          res_cont_vect[idx, ...,t_idx],
          torch.abs(res_cont[idx, ...,t_idx] - res_cont_vect[idx, ...,t_idx])
          ]

titles = [
          'ConvOps ',
          'Vector ConvOps',
          'Abs Error'
          ]

subplots_2d(values, titles)

# %%
# %% #Checking continuity 
res_mom = residual_momentum(vars)
res_mom_vect = residual_momentum_vector(vars)

#Plotting the error bars. 
idx = 5
t_idx = 10
values = [
          res_mom[idx, ...,t_idx],
          res_mom_vect[idx, ...,t_idx],
          torch.abs(res_mom[idx, ...,t_idx] - res_mom_vect[idx, ...,t_idx])
          ]

titles = [
          'ConvOps ',
          'Vector ConvOps',
          'Abs Error'
          ]

subplots_2d(values, titles)

# %%
