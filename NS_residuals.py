#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc. 
"""

# %%
configuration = {"Case": 'Navier-Stokes',
                 "Field": 'u, v, p, w',
                 "Model": 'FNO',
                 "Epochs": 500,
                 "Batch Size": 50,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GeLU',
                 "Physics Normalisation": 'No',
                 "Normalisation Strategy": 'Min-Max',
                 "T_in": 10,    
                 "T_out": 40,
                 "Step": 10,
                 "Width_time": 16, 
                 "Width_vars": 0,  
                 "Modes": 8,
                 "Variables":4, 
                 "Loss Function": 'LP',
                 "UQ": 'None', #None, Dropout
                 }
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
from Neural_PDE.Models.FNO import *
from Neural_PDE.Utils.processing_utils import * 
from Neural_PDE.Utils.training_utils import * 

from plot_tools import subplots_2d

# %% 
#Setting up locations. 
file_loc = os.getcwd()
# data_loc = file_loc + '/Data'
model_loc = file_loc + '/Weights'
plot_loc = file_loc + '/Plots'
#Setting up the seeds and devices
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% 
# Generating the data through simulation:
#Example Usage
N = 400 #Number of grid points
L = 1 #Domain Length
tStart = 0.0 #Starting time of the simulation
tEnd = 0.5 #Simulation ending time
dt = 0.001 #dt
nu = 0.001#kinematic viscosity
aa = 0.75#parametrisation of initial Vx 
bb = 0.75#parametrisation of initial Vx 
t_slice = 10 
x_slice = 4

from Neural_PDE.Numerical_Solvers.Navier_Stokes.NS_2D_spectral import * 

solver= Navier_Stokes_2d(N, tStart, tEnd, dt, nu, L, aa, bb)
u, v, p, w, x, t, err = solver.solve()

x = x[::x_slice]
dt = t=0.001*t_slice
u=u[::t_slice, ::x_slice, ::x_slice]
v=v[::t_slice, ::x_slice, ::x_slice]
p=p[::t_slice, ::x_slice, ::x_slice]
w=w[::t_slice, ::x_slice, ::x_slice]
# %% 
def stack_fields(variables):
    stack = []
    for var in variables:
        var = torch.tensor(var, dtype=torch.float32).unsqueeze(0) #Converting to Torch
        var = var.permute(0, 2, 3, 1) #Permuting to be BS, Nx, Ny, Nt
        stack.append(var)
    stack = torch.stack(stack, dim=1)
    return stack

vars = stack_fields([u,v,p,w])

field = ['u', 'v', 'p', 'w']

u_in = vars[...,:configuration['T_in']]
u_out = vars[...,configuration['T_in'] : configuration['T_in'] + configuration['T_out']]

# %% 
#Normalisation
norms = np.load(model_loc + '/FNO_Navier-Stokes_partial-crumble_norms.npz')
#Loading the Normaliation values
in_normalizer = MinMax_Normalizer(u_in)
in_normalizer.a = torch.tensor(norms['in_a'])
in_normalizer.b = torch.tensor(norms['in_b'])

out_normalizer = MinMax_Normalizer(u_out)
out_normalizer.a = torch.tensor(norms['in_a'])
out_normalizer.b = torch.tensor(norms['in_b'])

u_in = in_normalizer.encode(u_in)
u_out_encoded = out_normalizer.encode(u_out)

# %%
#Load the model. 
model = FNO_multi2d(configuration['T_in'], configuration['Step'], configuration['Modes'], configuration['Modes'], configuration['Variables'], configuration['Width_time'])
model.load_state_dict(torch.load(model_loc + '/FNO_Navier-Stokes_partial-crumble.pth', map_location='cpu'))

#Model Predictions.
pred_encoded, mse, mae = validation_AR(model, u_in, u_out_encoded, configuration['Step'], configuration['T_out'])

print('(MSE) Error: %.3e' % (mse))
print('(MAE) Error: %.3e' % (mae))

#Denormalising the predictions
pred = out_normalizer.decode(pred_encoded.to(device)).cpu()

# %% 
#Estimating the Residuals

def unstack_fields(field, axis, variable_names):
    fields = torch.split(field, 1, dim=axis)
    fields = [t.squeeze(axis) for t in fields]
    
    if len(fields) != len(variable_names):
        raise ValueError("Number of tensors and variable names should match.")
    
    variables = []
    for field in fields:
        variables.append(field.permute(0, 3, 1, 2))
    
    return variables

u, v, p, w = unstack_fields(pred, axis=1, variable_names=field)#Prediction
# u, v, p, w = unstack_fields(u_out, axis=1, variable_names=field)#Solution

dx = np.asarray(x[-1] - x[-2])
dy = dx
dt = dt

alpha = 1/dt*2
beta = 1/dx*2
gamma = 1/dx**2                                                                                                                                        

from ConvOps import ConvOperator
#Defining the required Convolutional Operations. 
D_t = ConvOperator(domain='t', order=1, scale=alpha)
D_x = ConvOperator(domain='x', order=1, scale=beta) 
D_y = ConvOperator('y', 1, beta )
D_x_y = ConvOperator(('x', 'y'), 1, beta)
D_xx_yy = ConvOperator(('x','y'), 2, gamma)

#Continuity
residual_cont = D_x(u) + D_y(v) 

#Momentum
residual_mom_x = D_t(u) + u*D_x(u) + v*D_y(u) - nu*D_xx_yy(u) + D_x(p)
residual_mom_y = D_t(v) + u*D_y(v) + v*D_y(v) - nu*D_xx_yy(v) + D_y(p)

# %% 
# Example values to plot
t_idx = -1
values = [residual_cont[t_idx][1:-1,1:-1], residual_mom_x[0, t_idx][1:-1,1:-1], residual_mom_y[0, t_idx][1:-1,1:-1]]
titles = ["Cont.", "Mom_X", "Mom_Y"]

subplots_2d(values, titles)

# %%
#############################################################################
#Performing the Inverse mapping from the Residuals to the Fields
#############################################################################

# u_integrate = D.integrate(u_val)

# values=[u_val[0, t_idx], u_integrate[0, t_idx], torch.abs(u_val[0, t_idx] - u_integrate[0, t_idx])]
# titles = ['Actual', 'Retrieved', 'Abs Diff']
# subplots_2d(values, titles)
# %% 
