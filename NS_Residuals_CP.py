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
data =  np.load(data_loc + '/NS_Spectral_combined.npz')

u = data['u'].astype(np.float32)
v = data['v'].astype(np.float32)
p = data['p'].astype(np.float32)
w = data['w'].astype(np.float32)
x = data['x'].astype(np.float32)
y = data['x'].astype(np.float32)
dt = data['dt'].astype(np.float32)
nu = 0.001#kinematic viscosity

# %% 
def stacked_fields(variables):
    stack = []
    for var in variables:
        var = torch.from_numpy(var) #Converting to Torch
        var = var.permute(0, 2, 3, 1) #Permuting to be BS, Nx, Ny, Nt
        stack.append(var)
    stack = torch.stack(stack, dim=1)
    return stack

vars = stacked_fields([u,v,p,w])

field = ['u', 'v', 'p', 'w']

n_cal = configuration['n_cal']
u_in = vars[:n_cal,...,:configuration['T_in']]
u_out = vars[:n_cal,...,configuration['T_in'] : configuration['T_in'] + configuration['T_out']]

# %% 
#Normalisation
norms = np.load(model_loc + '/FNO_Navier-Stokes_warm-dialogue_norms.npz')
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
model.load_state_dict(torch.load(model_loc + '/FNO_Navier-Stokes_warm-dialogue.pth', map_location='cpu'))

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

u, v, p, w = unstack_fields(pred, axis=1, variable_names=field)#Prediction 
# u, v, p, w = unstack_fields(u_out, axis=1, variable_names=field)#Solution

dx = np.asarray(x[-1] - x[-2])
dy = dx
dt = dt

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

#Continuity
# residual_cont = D_x(u) + D_y(v) 
residual_cont_cal = D_x(u) + (dx/dy) * D_y(v) #Rescaling

#Momentum
# residual_mom_x = D_t(u) + u*D_x(u) + v*D_y(u) - nu*D_xx_yy(u) + D_x(p)
residual_momx_cal = D_t(u)*dx*dy + u*D_x(u)*dt*dy + v*D_y(u)*dt*dx - nu*D_xx_yy(u)*dt + D_x(p)*dt*dy #Rescaling

# residual_mom_y = D_t(v) + u*D_y(v) + v*D_y(v) - nu*D_xx_yy(v) + D_y(p)
residual_momy_cal = D_t(v)*dx*dy + u*D_y(v)*dt*dx + v*D_y(v)*dt*dy - nu*D_xx_yy(v)*dt + D_y(p)*dt*dx #Rescaling

# %% 
# Example values to plot
idx = 0
t_idx = 5
values = [residual_cont_cal[idx, t_idx][1:-1,1:-1], residual_momx_cal[idx, t_idx][1:-1,1:-1], residual_momy_cal[idx, t_idx][1:-1,1:-1]]
titles = ["Cont.", "Mom_X", "Mom_Y"]

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

u, v, p, w = unstack_fields(pred, axis=1, variable_names=field)#Prediction
# u, v, p, w = unstack_fields(u_out, axis=1, variable_names=field)#Solution

#Continuity
# residual_cont = D_x(u) + D_y(v) 
residual_cont_pred = D_x(u) + (dx/dy) * D_y(v) #Rescaling

#Momentum
# residual_mom_x = D_t(u) + u*D_x(u) + v*D_y(u) - nu*D_xx_yy(u) + D_x(p)
residual_momx_pred = D_t(u)*dx*dy + u*D_x(u)*dt*dy + v*D_y(u)*dt*dx - nu*D_xx_yy(u)*dt + D_x(p)*dt*dy #Rescaling

# residual_mom_y = D_t(v) + u*D_y(v) + v*D_y(v) - nu*D_xx_yy(v) + D_y(p)
residual_momy_pred = D_t(v)*dx*dy + u*D_y(v)*dt*dx + v*D_y(v)*dt*dy - nu*D_xx_yy(v)*dt + D_y(p)*dt*dx #Rescaling

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

#Continuity
# residual_cont = D_x(u) + D_y(v) 
residual_cont_val = D_x(u) + (dx/dy) * D_y(v) #Rescaling
#Momentum
# residual_mom_x = D_t(u) + u*D_x(u) + v*D_y(u) - nu*D_xx_yy(u) + D_x(p)
residual_momx_val = D_t(u)*dx*dy + u*D_x(u)*dt*dy + v*D_y(u)*dt*dx - nu*D_xx_yy(u)*dt + D_x(p)*dt*dy #Rescaling
# residual_mom_y = D_t(v) + u*D_y(v) + v*D_y(v) - nu*D_xx_yy(v) + D_y(p)
residual_momy_val = D_t(v)*dx*dy + u*D_y(v)*dt*dx + v*D_y(v)*dt*dy - nu*D_xx_yy(v)*dt + D_y(p)*dt*dx #Rescaling

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