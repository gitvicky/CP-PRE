#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc. 

u_t + u*u_x =  nu*u_xx on [0,2]

"""

# %%
configuration = {"Case": 'Burgers',
                 "Field": 'u',
                 "Model": 'FNO',
                 "Epochs": 500,
                 "Batch Size": 50,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.001,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'Tanh',
                 "Normalisation Strategy": 'Identity',
                 "T_in": 20,    
                 "T_out": 30,
                 "Step": 30,
                 "Width": 32, 
                 "Modes": 8,
                 "Variables":1, 
                 "Noise":0.0, 
                 "Loss Function": 'MSE',
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

from Neural_PDE.Numerical_Solvers.Burgers.Burgers_1D import * 

Nx = 1000 #Number of x-points
Nt = 500 #Number of time instances 
x_min = 0.0 #Min of X-range 
x_max = 2.0 #Max of X-range 
t_end = 1.25 #Time Maximum
nu = 0.002
x_slice = 5
t_slice = 10

alpha, beta, gamma = 1.0, 1.0, 1.0

sim = Burgers_1D(Nx, Nt, x_min, x_max, t_end, nu) 
sim.InitializeU(alpha, beta, gamma)
u_sol, x, dt = sim.solve()

u_sol = u_sol[::t_slice, ::x_slice]
x = x[::x_slice]
dt = dt*t_slice

u = torch.tensor(u_sol, dtype=torch.float32)
u = u.unsqueeze(0).permute(0, 2, 1) #Adding BS and Permuting for FNO
u = u.unsqueeze(1) #Adding the variable channel

u_in = u[...,:configuration['T_in']]
u_out = u[...,configuration['T_in'] : configuration['T_in'] + configuration['T_out']]

# %% 
#Normalisation -- NA - Identity
'''
# norms = np.load(model_loc + '/FNO_Burgers_loud-mesh_norms.npz')
# #Loading the Normaliation values
# in_normalizer = MinMax_Normalizer(u_in)
# in_normalizer.a = torch.tensor(norms['in_a'])
# in_normalizer.b = torch.tensor(norms['in_b'])

# out_normalizer = MinMax_Normalizer(u_out)
# out_normalizer.a = torch.tensor(norms['in_a'])
# out_normalizer.b = torch.tensor(norms['in_b'])

# u_in = in_normalizer.encode(u_in)
# u_out_encoded = out_normalizer.encode(u_out)
'''
u_out_encoded = u_out
# %%
#Load the model. 
model = FNO_multi1d(configuration['T_in'], configuration['Step'], configuration['Modes'], configuration['Variables'], configuration['Width'], width_vars=0)
model.load_state_dict(torch.load(model_loc + '/FNO_Burgers_loud-mesh.pth', map_location='cpu'))

#Model Predictions.
pred_encoded, mse, mae = validation_AR(model, u_in, u_out_encoded, configuration['Step'], configuration['T_out'])

print('(MSE) Error: %.3e' % (mse))
print('(MAE) Error: %.3e' % (mae))

#Denormalising the predictions
# pred = out_normalizer.decode(pred_encoded.to(device)).cpu()
pred = pred_encoded
# %% 
#Estimating the Residuals

uu = pred #Prediction
uu = u_out #Solution

dx = np.asarray(x[-1] - x[-2])
dt = dt

alpha = 1/dt*2
beta = 1/dx*2
gamma = 1/dx**2                                                                                                                                        

from ConvOps import ConvOperator
#Defining the required Convolutional Operations. 
D_t = ConvOperator(domain='t', order=1, scale=alpha)
D_x = ConvOperator(domain='x', order=1, scale=beta) 
D_xx = ConvOperator(domain='x', order=2, scale=gamma)

# Residual
residual_cont = D_t(uu) + uu * D_x(uu) - nu * D_xx(uu)


# # %% 
# # Example values to plot
# t_idx = -1
# values = [residual_cont[t_idx][1:-1,1:-1], residual_mom_x[0, t_idx][1:-1,1:-1], residual_mom_y[0, t_idx][1:-1,1:-1]]
# titles = ["Cont.", "Mom_X", "Mom_Y"]

# subplots_2d(values, titles)

# # %%
# #############################################################################
# #Performing the Inverse mapping from the Residuals to the Fields
# #############################################################################

# # u_integrate = D.integrate(u_val)

# # values=[u_val[0, t_idx], u_integrate[0, t_idx], torch.abs(u_val[0, t_idx] - u_integrate[0, t_idx])]
# # titles = ['Actual', 'Retrieved', 'Abs Diff']
# # subplots_2d(values, titles)
# # %% 

# %%
