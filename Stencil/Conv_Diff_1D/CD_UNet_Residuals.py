#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

1D U-Net built using PyTorch to model the 1D Burgers Equation. 
Conformal Prediction using various Conformal Score estimates

"""

# %% 
configuration = {"Case": 'Conv-Diff',
                 "Field": 'u',
                 "Type": 'Unet',
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
                 "Physics Normalisation": 'No',
                 "T_in": 10,    
                 "T_out": 10,
                 "Step": 10,
                 "Width": 32, 
                 "Variables":1, 
                 "Noise":0.0, 
                 "Loss Function": 'Quantile Loss',
                 "UQ": 'None',
                 "Pinball Gamma": 0.95,
                 "Dropout Rate": 'NA',
                 "Spatial Resolution": 200
                 }

# %%
#Importing the necessary packages
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

from collections import OrderedDict
from utils import *

torch.manual_seed(0)
np.random.seed(0)
from utils import *


# %% 
#Setting the seeds and the path for the run. 
torch.manual_seed(0)
np.random.seed(0)
path = os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #positioning the GPUs if they are available. only works for Nvidia at the moment 

model_loc = path + '/Models/'
data_loc = path

# %% 

################################################################
# load data
# _a -- referes to the input 
# _u -- referes ot the output
################################################################
t1 = default_timer()

data =  np.load(data_loc + '/Data/Conv_Diff_u_2.npz')
u_sol = data['u'].astype(np.float32) [:, ::5, :]
u = torch.from_numpy(u_sol)

x_range =  np.arange(0, 10, 0.05)

ntrain = 3000
ncal = 1000
npred = 1000
S = 200 #Grid Size

width = configuration['Width']
output_size = configuration['Step']
batch_size = configuration['Batch Size']

T_in = configuration['T_in']
T = configuration['T_out']
step = configuration['Step']

# %%
#Chunking the data. 
u_train = torch.from_numpy(np.load(data_loc + '/Data/Conv_Diff_u_1.npz')['u'].astype(np.float32) [:, ::5, :])

train_a = u_train[:ntrain,:T_in,:]
train_u = u_train[:ntrain,T_in:T+T_in,:]

cal_a = u[:ncal,:T_in, :]
cal_u = u[:ncal,T_in:T+T_in,:]

pred_a = u[ncal:ncal+npred,:T_in, :]
pred_u = u[ncal:ncal+npred,T_in:T+T_in,:]

print(train_u.shape)
print(cal_u.shape)
print(pred_u.shape)

t2 = default_timer()
print('Data sorting finished, time used:', t2-t1)

# %% 
#Normalisation. 
a_normalizer = MinMax_Normalizer(train_a)
train_a = a_normalizer.encode(train_a)
cal_a = a_normalizer.encode(cal_a)
pred_a = a_normalizer.encode(pred_a)

y_normalizer = MinMax_Normalizer(train_u)
train_u = y_normalizer.encode(train_u)
cal_u = y_normalizer.encode(cal_u)
pred_u = y_normalizer.encode(pred_u)


# %%
#Loading the trained model. 
model = UNet1d(T_in, step, width)
model.load_state_dict(torch.load(model_loc + 'Unet_CD_mean_0.5.pth', map_location='cpu'))

# %%

#Generating Test Data through some more numerical runs and then testing it on the surrogate.
from pyDOE import lhs
from CD_numerical import *


dx = 0.05 #x discretisation
dt = 0.0005 #t discretisation
D_damp = 2*np.pi #Damping Coefficient of the spatial diffusion coefficient
c = 0.5 #Convection Velocity  
mu = 5.0  #Mean of the initial Gaussian
sigma = 0.5 #Variance of the initial Gaussian. 

solver = Conv_Diff_1d(dx, dt, D_damp, c, mu, sigma) 
u_sol, D, D_x = solver.solve()  #solution shape -> t, x

u_test = torch.tensor(u_sol, dtype=torch.float32)
u_test = u_test.unsqueeze(0)
u_test = u_test[:, ::5, :] #Sliced for every 5th as in the trained models. 

test_a = u_test[:,:T_in,:]
test_u = u_test[:,T_in:T+T_in,:]

#Normalising the test data 
test_a = a_normalizer.encode(test_a)
# test_u = y_normalizer.encode(test_u)

# %% 
#Obtaining the Predictions
with torch.no_grad():
    pred_set = model(test_a)

pred_set = y_normalizer.decode(pred_set)

# %% 
#Estimating the Residuals 

#Testing for a single simulation prediction only
#Residual Estimations across the spatio-temporal tensor

dx = 0.05
dt = 0.0005
slice = 5
dt = slice*dt #slice

u_tensor = torch.tensor(pred_set, dtype=torch.float32).reshape(1, 1, pred_set.shape[1], pred_set.shape[2])

alpha = (2*dt*D[0]/dx**2)
beta = (c*dt)/dx

stencil = torch.tensor([[0., -1., 0.],
                           [alpha-beta, -2*alpha, alpha+beta],
                           [0., 1., 0.]], dtype=torch.float32)
                           
stencil = stencil.view(1, 1, 3, 3)

deriv_stencil_conv = F.conv2d(u_tensor, stencil)[0,0]

# %% 
idx = 0
u_actual = test_u[idx].numpy()
u_pred = pred_set[idx].numpy()
u_mae = np.abs(u_actual - u_pred)
u_mse = (u_actual - u_pred)**2

u_t = np.gradient(u_pred, dt, edge_order=2, axis=0)
u_x = np.gradient(u_pred, dx, edge_order=2, axis=1)
u_xx = np.gradient(u_x, dx, edge_order=2, axis=1)

D_x = np.gradient(D[idx], dx, edge_order=2, axis=0)
c =params[idx, 1]
u_residual_individual = u_t - D[idx]*u_x - u_pred*D_x + c*u_x 

# grads = np.gradient(u_pred, dt, dx, axis=[0,1]) #Estimating gradients together
# u_t = grads[0]
# u_x = grads[1]
u_residual_individual = u_t - D[idx]*u_x - u_pred*D_x + c*u_x 
# grads = np.gradient(pred_set, dt, dx, axis=[1,2]) #Estimating gradients across the batch 
# u_residua_together = grads[0][:, :] + v[80+idx]*grads[1][:, :]

# %%
# 
# %% 
#Test Plots
from mpl_toolkits.axes_grid1 import make_axes_locatable

idx = 0
u_actual = test_u[idx].numpy()
u_pred = pred_set[idx].numpy()
u_mae = np.abs(u_actual - u_pred)
# u_pred = u_actual

fig = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.1)

ax = fig.add_subplot(2,2,1)
pcm =ax.imshow(u_actual, cmap=cm.coolwarm, extent=[0.0, 2.0, 0, 0.5])
ax.title.set_text('Actual')
ax.set_xlabel('x')
ax.set_ylabel('t')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(2,2,2)
pcm =ax.imshow(u_pred, cmap=cm.coolwarm, extent=[0.0, 2.0, 0, 0.5])
ax.title.set_text('Pred')
ax.set_xlabel('x')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))


#Absolute Error 
ax = fig.add_subplot(2,2,3)
pcm =ax.imshow(u_mae, cmap=cm.coolwarm, extent=[0.0, 2.0, 0, 0.5])
ax.title.set_text('Abs Error')
ax.set_xlabel('x')
ax.set_ylabel('t')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))


#Squared Error 
ax = fig.add_subplot(2,2,4)
pcm =ax.imshow(deriv_stencil_conv, cmap=cm.coolwarm, extent=[0.0, 2.0, 0, 0.5])
ax.title.set_text('Residual')
ax.set_xlabel('x')
# ax.set_ylabel('t')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))
