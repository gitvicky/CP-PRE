#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12th Dec 2023

FNO built using PyTorch to model the 2D Wave Equation. 
Dataset buitl by changing by performing a LHS across the x,y pos and amplitude of the initial gaussian distibution
Code for the spectral solver can be found in : https://github.com/farscape-project/PINNs_Benchmark

----------------------------------------------------------------------------------------------------------------------------------------

UQ by way of residuals. Residual estimation using Convolutional Kernels as Finite Difference Stencils
"""

# %%
configuration = {"Case": 'Wave',
                 "Field": 'u',
                 "Type": 'FNO',
                 "Epochs": 500,
                 "Batch Size": 50,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'Tanh',
                 "Normalisation Strategy": 'Min-Max',
                 "Instance Norm": 'No',
                 "Log Normalisation":  'No',
                 "Physics Normalisation": 'Yes',
                 "T_in": 20,    
                 "T_out": 60,
                 "Step": 10,
                 "Width": 32, 
                 "Modes": 8,
                 "Variables":1, 
                 "Noise":0.0, 
                 "Loss Function": 'LP',
                 "UQ": 'Dropout', #None, Dropout
                 "Pinball Gamma": 'NA',
                 "Dropout Rate": 0.1
                 }

#%% 

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

# %% 
#Settung up locations. 
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

Nx = 33 # Mesh Discretesiation 
Nt = 100 #Max time
x_min = -1.0 # Minimum value of x
x_max = 1.0 # maximum value of x
y_min = -1.0 # Minimum value of y 
y_max = 1.0 # Minimum value of y
tend = 1
Lambda = 20 #Gaussian Amplitude
aa = 0.25 #X-pos
bb = 0.25 #Y-pos
c = 1.0 #Wave Speed <=1.0

#Initialising the Solver
from Neural_PDE.Numerical_Solvers.Wave import Wave_2D_Spectral
solver = Wave_2D_Spectral.Wave_2D(Nx, Nt, x_min, x_max, tend, c, Lambda, aa , bb)

#Solving and obtaining the solution. 
x, y, t, u_sol = solver.solve() #solution shape -> t, x, y

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

u = torch.tensor(u_sol, dtype=torch.float32)#converting to torch 
u = u.permute(1, 2, 0)#BS, Nx, Ny, Ntx_min = -1.0 # Minimum value of x
x_max = 1.0 # maximum value of x
y_min = -1.0 # Minimum value of y 
y_max = 1.0 # Minimum value of y
tend = 1
u = u.unsqueeze(0).unsqueeze(1)#BS, vars, Nx, Ny, Nt

u_in = u[...,:configuration['T_in']]
u_out = u[...,configuration['T_in'] : configuration['T_in'] + configuration['T_out']]

# %% 
#Normalisation. - Setting up
in_normalizer = MinMax_Normalizer(u_in)
out_normalizer = MinMax_Normalizer(u_out)

u_in = in_normalizer.encode(u_in)
u_out_encoded = out_normalizer.encode(u_out)

# %%
#Load the model. 
model = FNO_multi( configuration['T_in'], configuration['Step'], configuration['Modes'], configuration['Modes'], configuration['Variables'], configuration['Width'])
model.load_state_dict(torch.load(model_loc + '/FNO_Wave_customer-watt.pth', map_location='cpu'))

#Model Predictions.
pred_encoded, mse, mae = validation_AR(model, u_in, u_out_encoded, configuration['Step'], configuration['T_out'])

print('(MSE) Testing Error: %.3e' % (mse))
print('(MAE) Testing Error: %.3e' % (mae))

#Denormalising the predictions
pred = out_normalizer.decode(pred_encoded.to(device)).cpu()


# %% 
#Estimating the Residuals 
# u_tt  = c**2 * (u_xx + u_yy)

u_val = u_out[:, 0] #Validating on Numerical Solution 
# u_val = pred[:, 0] #Prediction
u_val = u_val.permute(0, 3, 1, 2) #BS, Nt, Nx, Nt

dx = x[-1] - x[-2]
dy = y[-1] - y[-2]
dt = t[-1] - t[-2]

alpha = 1/dx**2
beta = 1/dt**2

from Conv_Derivs import DerivConv
#Defining the required Convolutional Operations. 
D_tt = DerivConv('t', 2)
D_xx_yy = DerivConv(('x','y'), 2)

u_tt = D_tt(u_val)[:, 1:-1,1:-1]
u_xx_yy = D_xx_yy(u_val)[:, 1:-1,1:-1]

u_residual = u_tt - (c*dt/dx)**2 * u_xx_yy

# %% 
from matplotlib import pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
fig = plt.figure(figsize=(20, 4))
idx = 10

# Selecting the axis-X making the bottom and top axes False. 
plt.tick_params(axis='x', which='both', bottom=False, 
                top=False, labelbottom=False) 
  
# Selecting the axis-Y making the right and left axes False 
plt.tick_params(axis='y', which='both', right=False, 
                left=False, labelleft=False) 
  # Remove frame
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

ax = fig.add_subplot(1,4,1)
pcm =ax.imshow(u_val[0,idx], cmap='jet', origin='lower', extent=[x_min, x_max, y_min, y_max])#, vmin=mini, vmax=maxi)
ax.title.set_text(r'$(u)$')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)


ax = fig.add_subplot(1,4,2)
pcm =ax.imshow(u_xx_yy[idx], cmap='jet', origin='lower', extent=[x_min, x_max, y_min, y_max])#, vmin=mini, vmax=maxi)
ax.title.set_text(r'$(u_{xx} + u_{yy})$')
ax.set_xlabel('x')
# ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(1,4,3)
pcm =ax.imshow(u_tt[idx], cmap='jet', origin='lower',extent=[x_min, x_max, y_min, y_max])#,  vmin=mini, vmax=maxi)
ax.title.set_text(r'$u_t$')
ax.set_xlabel('x')
# ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(1,4,4)
pcm =ax.imshow(u_residual[idx], cmap='jet', origin='lower',extent=[x_min, x_max, y_min, y_max])#,  vmin=mini, vmax=maxi)
ax.title.set_text(r'$Residual$')
ax.set_xlabel('x')
# ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

# %%