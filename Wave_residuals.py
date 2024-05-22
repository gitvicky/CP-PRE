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
                 "Step": 5,
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
c = 1.0 # Wave Speed <=1.0

#Initialising the Solver
from Neural_PDE.Numerical_Solvers.Wave import Wave_2D_Spectral
solver = Wave_2D_Spectral.Wave_2D(Nx, Nt, x_min, x_max, tend, c, Lambda, aa , bb)

#Solving and obtaining the solution. 
x, y, t, u_sol = solver.solve() #solution shape -> t, x, y

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

u = torch.tensor(u_sol, dtype=torch.float32)#converting to torch 
u = u.permute(1, 2, 0)#BS, Nx, Ny, Nt
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
# model.load_state_dict(torch.load(model_loc + 'FNO_Wave_fundamental-steak.pth', map_location='cpu'))

#Model Predictions.
pred_encoded, mse, mae = validation_AR(model, u_in, u_out_encoded, configuration['Step'], configuration['T_out'])

print('(MSE) Testing Error: %.3e' % (mse))
print('(MAE) Testing Error: %.3e' % (mae))

#Denormalising the predictions
pred = out_normalizer.decode(pred_encoded.to(device)).cpu()

# %% 
#Estimating the Residuals 

u_val = pred[:,0]
u_val = u_val.permute(0, 3, 1, 2) #BS, Nt, Nx, Nt

dx = x[-1] - x[-2]
dy = y[-1] - y[-2]
dt = t[-1] - t[-2]

alpha = 1/(dt**2)
beta = 1/(12*dx**2)

from fd_stencils import *

# %% 
#Standard

three_point_stencil  = torch.tensor([[0, 1, 0],
                           [0, -2, 0],
                           [0, 1, 0]], dtype=torch.float32)

stencil_time = torch.zeros(3,3,3)
stencil_t = alpha*three_point_stencil
                           
stencil_time[:, 1, :] = stencil_t

stencil_xx = torch.zeros(3,3,3)
stencil_x = beta * three_point_stencil.T
                           
stencil_xx[1,: , :] = stencil_x


stencil_yy = torch.zeros(3,3,3)
stencil_y = beta * three_point_stencil.T
                           
stencil_yy[:, :, 1] = stencil_y


stencil_time = stencil_time.view(1, 1, 3, 3, 3)
stencil_xx = stencil_xx.view(1, 1, 3, 3, 3)
stencil_yy =  stencil_yy.view(1, 1, 3, 3, 3)

# deriv_stencil_conv = F.conv3d(u_tensor, stencil_time)[0,0] - F.conv3d(u_tensor, stencil_xx)[0,0] - F.conv3d(u_tensor, stencil_yy)[0,0]
deriv_stencil_conv = F.conv3d(u_tensor, stencil_time, padding=1)[0,0] - F.conv3d(u_tensor, stencil_xx, padding=1)[0,0] - F.conv3d(u_tensor, stencil_yy, padding=1)[0,0]

deriv_stencil_conv = deriv_stencil_conv.permute(1,2,0)
# %% 
#Test Plots
from mpl_toolkits.axes_grid1 import make_axes_locatable

idx = 0
t_idx = -5

u_actual = pred_u[idx].numpy()
u_pred = pred_set[idx].numpy()
u_mae = np.abs(u_actual - u_pred)

fig = plt.figure()
mpl.rcParams['figure.figsize']=(14, 14)
# plt.subplots_adjust(left=0.1, rght=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.1)

fig = plt.figure()
ax = fig.add_subplot(2,2,1)
pcm =ax.imshow(u_actual[...,t_idx], cmap=cm.coolwarm, extent=[-1, 1, -1, 1])
ax.title.set_text('Actual')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(2,2,2)
pcm =ax.imshow(u_pred[...,t_idx], cmap=cm.coolwarm, extent=[-1, 1, -1, 1])
ax.title.set_text('Pred')
ax.set_xlabel('x')
# ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

#Absolute Error 
ax = fig.add_subplot(2,2,3)
pcm =ax.imshow(u_mae[...,t_idx], cmap=cm.coolwarm, extent=[-1, 1, -1, 1])
ax.title.set_text('Abs Error')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

#Residual Error 
ax = fig.add_subplot(2,2,4)
pcm =ax.imshow(deriv_stencil_conv[...,t_idx], cmap=cm.coolwarm, extent=[-1, 1, -1, 1])
ax.title.set_text('Residual')
ax.set_xlabel('x')
# ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

# %%
#Exploring the calibration method. 
# y = u_actual[...,t_idx]
# y_tilde = u_pred[...,t_idx]
# y_tilde_residual = deriv_stencil_conv[...,t_idx]
# # %%

# %%

# %%

