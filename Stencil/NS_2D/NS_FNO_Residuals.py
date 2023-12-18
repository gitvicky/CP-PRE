# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 6 Jan 2023
@author: vgopakum
FNO modelled over the MHD data built using JOREK for multi-blob diffusion.

Multivariable FNO
"""
# %%
configuration = {"Case": 'NS Turbulence Spectral',
                 "Field": 'u, v, w, p',
                 "Field_Mixing": 'Channel',
                 "Type": '2D Time',
                 "Epochs": 500,
                 "Batch Size": 5,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.001,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GELU',
                 "Normalisation Strategy": 'Min-Max. Same',
                 "Instance Norm": 'No',
                 "Log Normalisation": 'No',
                 "Physics Normalisation": 'Yes',
                 "T_in": 10,
                 "T_out": 40,
                 "Step":5,
                 "Modes": 16,
                 "Width_time": 32,  # FNO
                 "Width_vars": 0,  # U-Net
                 "Variables": 4,
                 "Noise": 0.0,
                 "Loss Function": 'LP Loss',
                 "Spatial Resolution": 1,
                 "Temporal Resolution": 1,
                 "Gradient Clipping Norm": None,
                 "Ntrain": 95
                 #  "UQ": 'Dropout',
                 #  "Dropout Rate": 0.9
                 }

# %%
from simvue import Run
run = Run(mode='disabled')
run.init(folder="/Conformal_Prediction", tags=['Conformal Prediction', 'MultiVariable', "Z_Li", "Skip-connect", "Diff", "Recon"], metadata=configuration)

# %%
import os
CODE = ['FNO_multiple_NS_Spectral.py']

# Save code files
for code_file in CODE:
    if os.path.isfile(code_file):
        run.save(code_file, 'code')
    elif os.path.isdir(code_file):
        run.save_directory(code_file, 'code', 'text/plain', preserve_path=True)
    else:
        print('ERROR: code file %s does not exist' % code_file)


# %%

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
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

# %%
path = os.getcwd()
data_loc = path + '/Data'
model_loc = path + '/Models'
file_loc = os.getcwd()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
# Extracting the configuration settings

modes = configuration['Modes']
width_time = configuration['Width_time']
width_vars = configuration['Width_vars']
output_size = configuration['Step']

batch_size = configuration['Batch Size']

batch_size2 = batch_size

t1 = default_timer()

T_in = configuration['T_in']
T = configuration['T_out']
step = configuration['Step']
num_vars = configuration['Variables']

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%

################################################################
# Loading Data
################################################################

# %%
data = data_loc + '/NS_Spectral_combined100.npz'
# %%
ield = configuration['Field']
dims = ['u', 'v', 'p', 'w']
num_vars = configuration['Variables']

u_sol = np.load(data)['u'].astype(np.float32) 
v_sol = np.load(data)['v'].astype(np.float32)
p_sol = np.load(data)['p'].astype(np.float32)
w_sol = np.load(data)['w'].astype(np.float32)

u_sol = np.nan_to_num(u_sol)
v_sol = np.nan_to_num(v_sol)
p_sol = np.nan_to_num(p_sol)
w_sol = np.nan_to_num(w_sol)

u = torch.from_numpy(u_sol)
u = u.permute(0, 2, 3, 1)

v = torch.from_numpy(v_sol)
v = v.permute(0, 2, 3, 1)

p = torch.from_numpy(p_sol)
p = p.permute(0, 2, 3, 1)

w = torch.from_numpy(w_sol)
w = w.permute(0, 2, 3, 1)

t_res = configuration['Temporal Resolution']
x_res = configuration['Spatial Resolution']
uvp = torch.stack((u, v, p, w), dim=1)[:, ::t_res]

x_grid = np.arange(0, 1, 0.0025)[::100]
y_grid =x_grid
t_grid = np.arange(0, len(u_sol), np.load(data)['dt'].astype(np.float32))


ntrain = configuration['Ntrain'] 
ntest = 5 

S = uvp.shape[3]  # Grid Size
size_x = S
size_y = S

batch_size = configuration['Batch Size']

batch_size2 = batch_size

t1 = default_timer()
print(uvp.shape)

train_a = uvp[:ntrain, :, :, :, :T_in]
train_u = uvp[:ntrain, :, :, :, T_in:T + T_in]

test_a = uvp[-ntest:, :, :, :, :T_in]
test_u = uvp[-ntest:, :, :, :, T_in:T + T_in]

print(train_u.shape)
print(test_u.shape)

# %%
a_normalizer = MinMax_Normalizer(uvp[...,:T_in])

train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

y_normalizer = MinMax_Normalizer(uvp[...,T_in:T+T_in])

train_u = y_normalizer.encode(train_u)
test_u_encoded = y_normalizer.encode(test_u)

# %%
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u_encoded), batch_size=batch_size,
                                          shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2 - t1)

# %%
################################################################
#Loading the model
################################################################
model = FNO_multi(modes, modes, width_time, T_in, step, num_vars, x_grid, y_grid)
model.load_state_dict(torch.load(model_loc + '/FNO_multi_blobs_resonnt-mayonnaise.pth', map_location='cpu'))
model.to(device)
model.eval()

run.update_metadata({'Number of Params': int(model.count_params())})

print("Number of model params : " + str(model.count_params()))

# optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'],
#                                             gamma=configuration['Scheduler Gamma'])

# myloss = LpLoss(size_average=False)

# %% 
#Generating test data 
from NS_Numerical import *
solver= Navier_Stokes_2d(400, 0.0, 0.5, 0.001, 0.001, 1, a=1, b=1) # N, tStart, tEnd, dt, nu, L
u, v, p, w, x, t, err = solver.solve()

# %% 
nu = 0.001
dt = 0.001
dx = x[-1] - x[-2]

t_slice = 10 
x_slice = 4

u=u[::t_slice, ::x_slice, ::x_slice]
v=v[::t_slice, ::x_slice, ::x_slice]
p=p[::t_slice, ::x_slice, ::x_slice]
w=w[::t_slice, ::x_slice, ::x_slice]
x=x[::x_slice]
dt=0.001*t_slice
# %% 

u_sol = torch.tensor(u, dtype=torch.float32).unsqueeze(0).permute(0, 2, 3, 1)
v_sol = torch.tensor(v, dtype=torch.float32).unsqueeze(0).permute(0, 2, 3, 1)
p_sol = torch.tensor(p, dtype=torch.float32).unsqueeze(0).permute(0, 2, 3, 1)
w_sol = torch.tensor(w, dtype=torch.float32).unsqueeze(0).permute(0, 2, 3, 1)

uvp = torch.stack((u_sol, v_sol, p_sol, w_sol), dim=1)[:, ::t_res]

test_a = uvp[:,:,:,:,:T_in]
test_u = uvp[:,:,:,:,T_in:T+T_in]

test_a = a_normalizer.encode(test_a)
test_u_encoded = y_normalizer.encode(test_u)

# %% 
# Testing using the test dataset
batch_size = 1
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u_encoded), batch_size=1,
                                          shuffle=False)
pred_set = torch.zeros(test_u.shape)
index = 0
with torch.no_grad():
    for xx, yy in tqdm(test_loader):
        loss = 0
        xx, yy = xx.to(device), yy.to(device)
        t1 = default_timer()
        for t in range(0, T, step):
            y = yy[..., t:t + step]
            out = model(xx)

            if t == 0:
                pred = out
            else:
                pred = torch.cat((pred, out), -1)

            xx = torch.cat((xx[..., step:], out), dim=-1)

        t2 = default_timer()
        pred_set[index] = pred
        index += 1
        print(t2 - t1)

# %%
print(pred_set.shape, test_u.shape)
# Logging Metrics
MSE_error = (pred_set - test_u_encoded).pow(2).mean()
MAE_error = torch.abs(pred_set - test_u_encoded).mean()

print('(MSE) Testing Error: %.3e' % (MSE_error))
print('(MAE) Testing Error: %.3e' % (MAE_error))


# %% 

pred_set_encoded = pred_set
pred_set = y_normalizer.decode(pred_set.to(device)).cpu()

# %%
# Plotting the comparison plots

idx = np.random.randint(0, ntest)
idx = 0

if configuration['Log Normalisation'] == 'Yes':
    test_u = torch.exp(test_u)
    pred_set = torch.exp(pred_set)

# %%
output_plot = []
for dim in range(num_vars):
    u_field = test_u[idx]

    v_min_1 = torch.min(u_field[dim, :, :, 0])
    v_max_1 = torch.max(u_field[dim, :, :, 0])

    v_min_2 = torch.min(u_field[dim, :, :, int(T / 2)])
    v_max_2 = torch.max(u_field[dim, :, :, int(T / 2)])

    v_min_3 = torch.min(u_field[dim, :, :, -1])
    v_max_3 = torch.max(u_field[dim, :, :, -1])

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(2, 3, 1)
    pcm = ax.imshow(u_field[dim, :, :, 0], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
    # ax.title.set_text('Initial')
    ax.title.set_text('t=' + str(T_in))
    ax.set_ylabel('Solution')
    fig.colorbar(pcm, pad=0.05)

    ax = fig.add_subplot(2, 3, 2)
    pcm = ax.imshow(u_field[dim, :, :, int(T / 2)], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2,
                    vmax=v_max_2)
    # ax.title.set_text('Middle')
    ax.title.set_text('t=' + str(int((T + T_in) / 2)))
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)

    ax = fig.add_subplot(2, 3, 3)
    pcm = ax.imshow(u_field[dim, :, :, -1], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
    # ax.title.set_text('Final')
    ax.title.set_text('t=' + str(T + T_in))
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)

    u_field = pred_set[idx]

    ax = fig.add_subplot(2, 3, 4)
    pcm = ax.imshow(u_field[dim, :, :, 0], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
    ax.set_ylabel('FNO')

    fig.colorbar(pcm, pad=0.05)

    ax = fig.add_subplot(2, 3, 5)
    pcm = ax.imshow(u_field[dim, :, :, int(T / 2)], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2,
                    vmax=v_max_2)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)

    ax = fig.add_subplot(2, 3, 6)
    pcm = ax.imshow(u_field[dim, :, :, -1], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)

    plt.title(dims[dim])

    # output_plot.append(file_loc + '/Plots/MultiBlobs_' + dims[dim] + '_' + run.name + '.png')
    # plt.savefig(output_plot[dim])

# %%

# INPUTS = []
# OUTPUTS = [model_loc, output_plot[0], output_plot[1], output_plot[2]]


# # Save input files
# for input_file in INPUTS:
#     if os.path.isfile(input_file):
#         run.save(input_file, 'input')
#     elif os.path.isdir(input_file):
#         run.save_directory(input_file, 'input', 'text/plain', preserve_path=True)
#     else:
#         print('ERROR: input file %s does not exist' % input_file)


# # Save output files
# for output_file in OUTPUTS:
#     if os.path.isfile(output_file):
#         run.save(output_file, 'output')
#     elif os.path.isdir(output_file):
#         run.save_directory(output_file, 'output', 'text/plain', preserve_path=True)   
#     else:
#         print('ERROR: output file %s does not exist' % output_file)

run.close()

# %%

#Utilising the stencil by way of convolutions using pytorch 
import torch.nn.functional as F

u = pred_set[idx][0]
v = pred_set[idx][1]
p = pred_set[idx][2]
w = pred_set[idx][3]


u_tensor =torch.tensor(u, dtype=torch.float32).reshape(1,1, u.shape[0], u.shape[1], u.shape[2])
v_tensor =torch.tensor(v, dtype=torch.float32).reshape(1,1, v.shape[0], v.shape[1], v.shape[2])
p_tensor =torch.tensor(p, dtype=torch.float32).reshape(1,1, p.shape[0], p.shape[1], p.shape[2])

alpha = 1/dt*2
beta = 1/dx*2
gamma = 1/dx**2

stencil_t = torch.zeros(3,3,3)
stencil = alpha*torch.tensor([[0, -1, 0],
                           [0, 0, 0],
                           [0, 1, 0]], dtype=torch.float32)
                           
stencil_t[:, 1, :] = stencil


stencil_x = torch.zeros(3,3,3)
stencil = beta * torch.tensor([[0, 0, 0],
                           [-1, 0 , 1],
                           [0, 0, 0]], dtype=torch.float32)
                           
stencil_x[1,: , :] = stencil

stencil_y = torch.zeros(3,3,3)
stencil = beta * torch.tensor([[0, 0, 0],
                           [-1, 0 , 1],
                           [0, 0, 0]], dtype=torch.float32)

stencil_y[:, :, 1] = stencil

# #Combining the x and y gradients together. 
# stencil_xy = torch.zeros(3,3,3)
# stencil_xy[1, :, :] = stencil
# stencil_xy[:, :, 1] = stencil

stencil_xx = torch.zeros(3,3,3)
stencil= gamma * torch.tensor([[0, 0, 0],
                           [1, -2 , 1],
                           [0, 0, 0]], dtype=torch.float32)
                           
stencil_xx[1,: , :] = stencil


stencil_yy = torch.zeros(3,3,3)
stencil = gamma * torch.tensor([[0, 0, 0],
                           [1, -2 , 1],
                           [0, 0, 0]], dtype=torch.float32)
                           
stencil_yy[:, :, 1] = stencil


nine_point_stencil  = torch.tensor([
                           [1, 1, 1],
                           [1, -8, 1],
                           [1, 1, 1]], dtype=torch.float32)


stencil_xx_yy = torch.zeros(3,3,3)
stencil = gamma * nine_point_stencil
stencil_xx_yy[1,: , :] = stencil

stencil_t = stencil_t.view(1, 1, 3, 3, 3)
stencil_x = stencil_x.view(1, 1, 3, 3, 3)
stencil_y =  stencil_y.view(1, 1, 3, 3, 3)
stencil_xx = stencil_xx.view(1, 1, 3, 3, 3)
stencil_yy =  stencil_yy.view(1, 1, 3, 3, 3)
stencil_xx_yy =  stencil_xx_yy.view(1, 1, 3, 3, 3)

deriv_u = F.conv3d(u_tensor, stencil_t)[0,0] + F.conv3d(u_tensor, stencil_x)[0,0]*u_tensor[...,1:-1, 1:-1, 1:-1] + F.conv3d(u_tensor, stencil_x)[0,0]*v_tensor[...,1:-1, 1:-1, 1:-1]  - nu*(F.conv3d(u_tensor, stencil_xx_yy)[0,0]) + F.conv3d(p_tensor, stencil_x)[0,0]
deriv_v = F.conv3d(u_tensor, stencil_t)[0,0] + F.conv3d(v_tensor, stencil_x)[0,0]*u_tensor[...,1:-1, 1:-1, 1:-1] + F.conv3d(v_tensor, stencil_x)[0,0]*v_tensor[...,1:-1, 1:-1, 1:-1]  - nu*(F.conv3d(v_tensor, stencil_xx_yy)[0,0]) + F.conv3d(p_tensor, stencil_y)[0,0]

deriv_cont = F.conv3d(u_tensor, stencil_x)[0,0] + F.conv3d(v_tensor, stencil_y)
deriv_u = deriv_u
deriv_v = deriv_v

deriv_stencil_conv = torch.cat((deriv_u, deriv_v, deriv_v), dim=1)
# %%
#Test Plots
from mpl_toolkits.axes_grid1 import make_axes_locatable

idx = 0
t_idx = -1

u_actual = test_u[idx].numpy()
u_pred = pred_set[idx].numpy()
u_mae = np.abs(u_actual - u_pred)
u_residual = deriv_stencil_conv[idx].numpy()

for ii in range(num_vars):
    fig = plt.figure()
    mpl.rcParams['figure.figsize']=(12, 12)

    fig = plt.figure()
    ax = fig.add_subplot(2,2,1)
    pcm =ax.imshow(u_actual[ii,...,t_idx], cmap=cm.coolwarm, extent=[-1, 1, -1, 1])
    ax.title.set_text('Actual')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))

    ax = fig.add_subplot(2,2,2)
    pcm =ax.imshow(u_pred[ii,...,t_idx], cmap=cm.coolwarm, extent=[-1, 1, -1, 1])
    ax.title.set_text('Pred')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))

    #Absolute Error 
    ax = fig.add_subplot(2,2,3)
    pcm =ax.imshow(u_mae[ii,...,t_idx], cmap=cm.coolwarm, extent=[-1, 1, -1, 1])
    ax.title.set_text('Abs Error')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))

    #Residual Error 
    ax = fig.add_subplot(2,2,4)
    pcm =ax.imshow(u_residual[ii,...,t_idx], cmap=cm.coolwarm, extent=[-1, 1, -1, 1])
    ax.title.set_text('Residual')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))

# %%