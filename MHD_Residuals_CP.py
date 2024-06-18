#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc. 
"""

# %%
configuration = {"Case": 'MHD',
                 "Field": 'rho, u, v, p, Bx, By',
                 "Model": 'FNO',
                 "Epochs": 500,
                 "Batch Size": 20,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GeLU',
                 "Physics Normalisation": 'No',
                 "Normalisation Strategy": 'Min-Max',
                 "T_in": 10,    
                 "T_out": 20,
                 "Step": 5,
                 "Width_time": 16, 
                 "Width_vars": 0,  
                 "Modes": 8,
                 "Variables":6, 
                 "Loss Function": 'LP',
                 "UQ": 'None', #None, Dropout
                "n_cal": 100, 
                 "n_pred": 100
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
data_loc = file_loc + '/Neural_PDE/Data'
model_loc = file_loc + '/Weights'
plot_loc = file_loc + '/Plots'
#Setting up the seeds and devices
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% 
# Loading the Calibration Data
t1 = default_timer()
data =  np.load(data_loc + '/Constrained_MHD_combined.npz')

rho = data['rho'].astype(np.float32)
u = data['v'].astype(np.float32)
v = data['v'].astype(np.float32)
p = data['p'].astype(np.float32)
Bx = data['Bx'].astype(np.float32)
By  = data['By'].astype(np.float32)

x = data['x'].astype(np.float32)
y = data['x'].astype(np.float32)
t = data['t'].astype(np.float32)

def stacked_fields(variables):
    stack = []
    for var in variables:
        var = torch.from_numpy(var) #Converting to Torch
        var = var.permute(0, 2, 3, 1) #Permuting to be BS, Nx, Ny, Nt
        stack.append(var)
    stack = torch.stack(stack, dim=1)
    return stack

x_slice = 1 
vars = stacked_fields([rho, u, v, p, Bx, By])[:, :, ::x_slice, ::x_slice, :]
field = ['rho', 'u', 'v', 'p', 'Bx', 'By']

n_cal = configuration['n_cal']
u_in = vars[:n_cal,...,:configuration['T_in']]
u_out = vars[:n_cal,...,configuration['T_in'] : configuration['T_in'] + configuration['T_out']]

# %% 
#Normalisation
norms = np.load(model_loc + '/FNO_MHD_parallel-easement_norms.npz')
#Loading the Normaliation values
in_normalizer = MinMax_Normalizer(u_in)
in_normalizer.a = torch.tensor(norms['in_a'])
in_normalizer.b = torch.tensor(norms['in_b'])

out_normalizer = MinMax_Normalizer(u_out)
out_normalizer.a = torch.tensor(norms['out_a'])
out_normalizer.b = torch.tensor(norms['out_b'])

u_in = in_normalizer.encode(u_in)
u_out_encoded = out_normalizer.encode(u_out)

# %%
#Load the model and Obtain predictions
model = FNO_multi2d(configuration['T_in'], configuration['Step'], configuration['Modes'], configuration['Modes'], configuration['Variables'], configuration['Width_time'])
model.load_state_dict(torch.load(model_loc + '/FNO_MHD_parallel-easement.pth', map_location='cpu'))

#Model Predictions.
pred_encoded, mse, mae = validation_AR(model, u_in, u_out_encoded, configuration['Step'], configuration['T_out'])

print('Calibration (MSE) Error: %.3e' % (mse))
print('Calibration (MAE) Error: %.3e' % (mae))

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

rho, u, v, p, Bx, By = unstack_fields(pred, axis=1, variable_names=field)#Prediction 
# rho, u, v, p, Bx, By= unstack_fields(u_out, axis=1, variable_names=field)#Solution

dx = np.asarray(x[-1] - x[-2])
dy = dx
dt = np.asarray(t[-1] - t[-2])

alpha = 1/dt*2
beta = 1/dx*2
gamma = 1/dx**2                 

alpha, beta, gamma = 1,1,1

from ConvOps import ConvOperator
#Defining the required Convolutional Operations. 
D_t = ConvOperator(domain='t', order=1)#, scale=alpha)
D_x = ConvOperator(domain='x', order=1)#, scale=beta) 
D_y = ConvOperator(domain='y', order=1)#, scale=beta)
D_x_y = ConvOperator(domain=('x', 'y'), order=1)#, scale=beta)
D_xx_yy = ConvOperator(domain=('x','y'), order=2)#, scale=gamma)

# Residual Estimation

# %% 
# Example values to plot
idx = 0
t_idx = 5
values = [u[idx, t_idx], v[idx, t_idx], 
          rho[idx, t_idx], p[idx, t_idx],
          Bx[idx, t_idx], By[idx, t_idx]]
titles = ["u", "v", 
          "rho", "p",
          "bx", "by"]

subplots_2d(values, titles)

# %% 
#Obtaining the predictions 
n_pred = configuration['n_pred']
u_in = vars[n_cal:n_cal+n_pred,...,:configuration['T_in']]
u_out = vars[n_cal:n_cal+n_pred,...,configuration['T_in'] : configuration['T_in'] + configuration['T_out']]

#Normalisations
u_in = in_normalizer.encode(u_in)
u_out_encoded = out_normalizer.encode(u_out)

#Model Predictions.
pred_encoded, mse, mae = validation_AR(model, u_in, u_out_encoded, configuration['Step'], configuration['T_out'])

print('Prediction (MSE) Error: %.3e' % (mse))
print('Prediction (MAE) Error: %.3e' % (mae))

#Denormalising the predictions
pred = out_normalizer.decode(pred_encoded.to(device)).cpu()

rho, u, v, p, Bx, By = unstack_fields(pred, axis=1, variable_names=field)#Prediction 
# rho, u, v, p, Bx, By= unstack_fields(u_out, axis=1, variable_names=field)#Solution

# Residual Estimation

# %% 
#Performing CP over the residual space
from Neural_PDE.UQ.inductive_cp import * 
ncf_scores = np.abs(residual_cont_cal.numpy())# / (uu_cal.numpy() + 1e-6) #MINE

alpha = 0.1
qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
prediction_sets = [residual_cont_pred.numpy() - qhat, residual_cont_pred.numpy() + qhat] 

# %% 
# #Checking for coverage:
#Obtaining the residuals for the Numerical Solution. 
u, v, p, w = unstack_fields(u_out, axis=1, variable_names=field)#Solution

#Residual Estimation 

#Emprical Coverage for all values of alpha 
alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov_res = []
for alpha in tqdm(alpha_levels):
    qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
    prediction_sets = [residual_cont_pred.numpy() - qhat, residual_cont_pred.numpy() + qhat]
    # prediction_sets = [pred_residual.numpy() - qhat*uu_pred.numpy(), pred_residual.numpy() + qhat*uu_pred.numpy()] #MINE
    emp_cov_res.append(emp_cov(prediction_sets, residual_cont_val.numpy()))

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()
# %% 