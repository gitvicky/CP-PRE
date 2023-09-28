#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

FNO built using PyTorch to model the 2D Wave Equation. 
Dataset buitl by changing by performing a LHS across the x,y pos and amplitude of the initial gaussian distibution
Code for the spectral solver can be found in : https://github.com/farscape-project/PINNs_Benchmark

----------------------------------------------------------------------------------------------------------------------------------------

Experimenting with a range of UQ Methods:
    1. Dropout
    2. Quantile Regression 
    3. NN Ensemble 
    4. Physics-based residual
    5. Deep Kernel Learning

Once UQ methodolgies have been demonstrated on each, we can use Conformal Prediction over a
 multitude of conformal scores to find empirically rigorous coverage. 
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

cal_a = u[ntrain:ntrain+ncal,:,:,:T_in]
cal_u = u[ntrain:ntrain+ncal,:,:,T_in:T+T_in]

pred_a = u[ntrain+ncal:ntrain+ncal+npred,:,:,:T_in]
pred_u = u[ntrain+ncal:ntrain+ncal+npred,:,:,T_in:T+T_in]


print(train_u.shape)
print(cal_u.shape)
print(pred_u.shape)

t2 = default_timer()
print('Data sorting finished, time used:', t2-t1)

# %% 
#Normalisation. 
a_normalizer = MinMax_Normalizer(train_a)
# train_a = a_normalizer.encode(train_a)
# cal_a = a_normalizer.encode(cal_a)
# pred_a = a_normalizer.encode(pred_a)

y_normalizer = MinMax_Normalizer(train_u)
# train_u = y_normalizer.encode(train_u)
# cal_u = y_normalizer.encode(cal_u)
# pred_u = y_normalizer.encode(pred_u)



# %%
#Load the model. 
model = FNO2d(modes, modes, width, T_in, step, x, y)
model.load_state_dict(torch.load(model_loc + 'FNO_Wave.pth', map_location='cpu'))

# %%
#Generating test simulation data 
from Spectral_Wave_Data_Gen import *
n_sims = 1
x, y, t, u_dataset = LHS_Sampling(n_sims)

u_dataset = np.asarray(u_dataset)
u_test = torch.tensor(u_dataset, dtype=torch.float32)

# u_test = u_test[:, ::5, :] #Sliced for every 5th as in the trained models. 

test_a = u_test[:,:T_in,:]
test_u = u_test[:,T_in:T+T_in,:]

test_a  = test_a.permute(0, 2, 3, 1)
test_u = test_u.permute(0, 2, 3, 1)

#Normalising the test data 
test_a = a_normalizer.encode(test_a)
# test_u = y_normalizer.encode(test_u)

# %% 
#Obtaining the Predictions
pred_set = torch.zeros(test_u.shape)
index = 0
with torch.no_grad():
        xx = test_a
        xx= xx.to(device)
        t1 = default_timer()
        for tt in range(0, T, step):
            out = model(xx)

            if tt == 0:
                pred = out
            else:
                pred = torch.cat((pred, out), -1)       

            xx = torch.cat((xx[..., step:], out), dim=-1)

        t2 = default_timer()
        pred_set[index]=pred
        index += 1

pred_set = y_normalizer.decode(pred_set)

# %% 
#Estimating the Residuals 

#Testing for a single simulation prediction only
#Residual Estimations across the spatio-temporal tensor
dt = t[1]
dx = x[-1] - x[-2]
dy = y[-1] - y[-2]


idx = 0
u_actual = test_u[idx].numpy()
u_pred = pred_set[idx].numpy()
u_mae = np.abs(u_actual - u_pred)
u_mse = (u_actual - u_pred)**2

u_t = np.gradient(u_pred, dt, edge_order=2, axis=-1)
u_x = np.gradient(u_pred, dx, edge_order=2, axis=1)
u_xx = np.gradient(u_x, dx, edge_order=2, axis=1)
u_y = np.gradient(u_pred, dy, edge_order=2, axis=2)
u_yy = np.gradient(u_y, dy, edge_order=2, axis=2)

u_residual_individual = u_t - u_xx - u_yy

# grads = np.gradient(u_pred, dt, dx, axis=[0,1]) #Estimating gradients together
# u_t = grads[0]
# u_x = grads[1]
# u_residual_individual = u_t - D[idx]*u_x - u_pred*D_x + c*u_x 
# grads = np.gradient(pred_set, dt, dx, axis=[1,2]) #Estimating gradients across the batch 
# u_residua_together = grads[0][:, :] + v[80+idx]*grads[1][:, :]

# %%
# 
# %% 
#Test Plots
from mpl_toolkits.axes_grid1 import make_axes_locatable

idx = 0
t_idx = 30

u_actual = test_u[idx].numpy()
u_pred = pred_set[idx].numpy()
u_mae = np.abs(u_actual - u_pred)
u_mse = (u_actual - u_pred)**2
# u_pred = u_actual

u_pred = u_pred + np.random.uniform(-1,1, u_pred.shape) * 1e-2

fig = plt.figure()
mpl.rcParams['figure.figsize']=(12, 12)
# plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.1)

ax = fig.add_subplot(3,2,1)
pcm =ax.imshow(u_actual[...,t_idx], cmap=cm.coolwarm, extent=[-1, 1, -1, 1])
ax.title.set_text('Actual')
ax.set_xlabel('x')
ax.set_ylabel('t')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(3,2,2)
pcm =ax.imshow(u_pred[...,t_idx], cmap=cm.coolwarm, extent=[-1, 1, -1, 1])
ax.title.set_text('Pred')
ax.set_xlabel('x')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))


#Absolute Error 
ax = fig.add_subplot(3,2,3)
pcm =ax.imshow(u_mae[...,t_idx], cmap=cm.coolwarm, extent=[-1, 1, -1, 1])
ax.title.set_text('Abs Error')
ax.set_xlabel('x')
ax.set_ylabel('t')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))


#Squared Error 
ax = fig.add_subplot(3,2,4)
pcm =ax.imshow(u_mse[...,t_idx], cmap=cm.coolwarm, extent=[-1, 1, -1, 1])
ax.title.set_text('L2 Error')
ax.set_xlabel('x')
# ax.set_ylabel('t')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

#First order FD
u_t = np.gradient(u_pred, dt, edge_order=1, axis=-1)
u_x = np.gradient(u_pred, dx, edge_order=1, axis=1)
u_xx = np.gradient(u_x, dx, edge_order=1, axis=1)
u_y = np.gradient(u_pred, dy, edge_order=1, axis=2)
u_yy = np.gradient(u_y, dy, edge_order=1, axis=2)

u_residual= u_t - u_xx - u_yy
u_residual = u_residual #/ 1e2

ax = fig.add_subplot(3,2,5)
pcm =ax.imshow(u_residual[...,t_idx], cmap=cm.coolwarm, extent=[-1, 1, -1, 1])
ax.title.set_text('1st Order FD')
ax.set_xlabel('x')
ax.set_ylabel('t')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

#Second Order FD 
u_t = np.gradient(u_pred, dt, edge_order=2, axis=-1)
u_x = np.gradient(u_pred, dx, edge_order=2, axis=1)
u_xx = np.gradient(u_x, dx, edge_order=2, axis=1)
u_y = np.gradient(u_pred, dy, edge_order=2, axis=2)
u_yy = np.gradient(u_y, dy, edge_order=2, axis=2)

u_residual = u_t - u_xx - u_yy

ax = fig.add_subplot(3,2,6)
pcm =ax.imshow(u_residual[...,t_idx], cmap=cm.coolwarm, extent=[-1, 1, -1, 1])
ax.title.set_text('2nd Order FD')
ax.set_xlabel('x')
# ax.set_ylabel('t')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

# %%


