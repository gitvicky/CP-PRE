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
                 "T_out": 40,
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
import os
import numpy as np
from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib import cm 
import matplotlib as mpl 

import operator
from functools import reduce
from functools import partial
from collections import OrderedDict

import time 
from timeit import default_timer
from tqdm import tqdm 

from utils import *

torch.manual_seed(0)
np.random.seed(0)

# %% 
path = os.getcwd()
model_loc = path + '/Models/'
data_loc = path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
################################################################
# load data
# _a -- referes to the input 
# _u -- referes ot the output
################################################################
t1 = default_timer()

data =  np.load(data_loc + '/Data/Spectral_Wave_data_LHS_5K.npz')
u_sol = data['u'].astype(np.float32)
x = data['x'].astype(np.float32)
y = data['y'].astype(np.float32)
t = data['t'].astype(np.float32)
u = torch.from_numpy(u_sol)
u = u.permute(0, 2, 3, 1)
xx, yy = np.meshgrid(x,y)

ntrain = 1000
ncal = 1000
npred = 1000


S = 33 #Grid Size

width = configuration['Width']
output_size = configuration['Step']
batch_size = configuration['Batch Size']

T_in = configuration['T_in']
T = configuration['T_out']
step = configuration['Step']
modes = configuration['Modes']
width = configuration['Width']
output_size = configuration['Step']


# %%
#Chunking the data. 
train_a = u[:ntrain,:,:,:T_in]
train_u = u[:ntrain,:,:,T_in:T+T_in]

t2 = default_timer()
print('Data sorting finished, time used:', t2-t1)

# %% 
#Normalisation. - Setting up
a_normalizer = MinMax_Normalizer(train_a)
y_normalizer = MinMax_Normalizer(train_u)


# %%
#Load the model. 
model = FNO_multi(modes, modes, width, T_in, step, 1,  x, y)
# model.load_state_dict(torch.load(model_loc + 'FNO_Wave.pth', map_location='cpu'))
model.load_state_dict(torch.load(model_loc + 'FNO_Wave_fundamental-steak.pth', map_location='cpu'))

# %%
# Generating test simulation data 
# Obtaining the numerical solution of the 1D Conv-Diff Equation 
from Wave_numerical import *

Lambda = 20 #Amplitude of the initial Gaussian. 
a = 0.25 #x-position of initial gaussian
b = -0.25 #y-position of initial gaussian 

solver = Wave_2D(Lambda, a , b)
xx, yy, t, u_sol = solver.solve() #solution shape -> t, x, y

dx = xx[-1] - xx[-2]
dy = yy[-1] - yy[-2]
dt = t[-1] - t[-2]

u_sol = torch.tensor(u_sol, dtype=torch.float32).unsqueeze(0).permute(0, 2, 3, 1)
pred_a = u_sol[:, ]

pred_a = u_sol[:,:,:,:T_in]
pred_u = u_sol[:,:,:,T_in:T+T_in]


# %% 
#Normalising the inputs 
pred_a = y_normalizer.encode(pred_a)

# %%
pred_a = pred_a.unsqueeze(1)
pred_u = pred_u.unsqueeze(1)
# %% 
#Obtaining the Predictions
index = 0
with torch.no_grad():
        xx = pred_a
        xx= xx.to(device)
        t1 = default_timer()
        for tt in tqdm(range(0, T, step)):
            out = model(xx)
            if tt == 0:
                pred = out
            else:
                pred = torch.cat((pred, out), -1)       

            xx = torch.cat((xx[..., step:], out), dim=-1)

        t2 = default_timer()
        pred_set=pred
        index += 1

pred_set = y_normalizer.decode(pred_set)

# %% 
#Estimating the Residuals 

pred_a = pred_a[:,0]
pred_u = pred_u[:,0]
pred_set = pred_set[:, 0]

u_val = pred_set.permute(0, 3, 1, 2)
u_tensor =torch.tensor(u_val, dtype=torch.float32).reshape(u_val.shape[0], 1, u_val.shape[1], u_val.shape[2], u_val.shape[3])

alpha = 1/(dt**2)
beta = 1/(12*dx**2)

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
deriv_stencil_conv = F.conv3d(u_tensor, stencil_time, padding=0)[0,0] - F.conv3d(u_tensor, stencil_xx, padding=0)[0,0] - F.conv3d(u_tensor, stencil_yy, padding=0)[0,0]

deriv_stencil_conv = deriv_stencil_conv.permute(1,2,0)
# %% 
#Test Plots
from mpl_toolkits.axes_grid1 import make_axes_locatable

idx = 0
t_idx = -5

u_actual = pred_u[idx][...,1:-1, 1:-1, 1:-1].numpy()
u_pred = pred_set[idx][...,1:-1, 1:-1, 1:-1].numpy()
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
y = u_actual[...,t_idx]
y_tilde = u_pred[...,t_idx]
y_tilde_residual = deriv_stencil_conv[...,t_idx]
# %%
