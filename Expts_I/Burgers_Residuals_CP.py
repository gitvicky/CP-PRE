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
import sys
sys.path.append("..")
from Neural_PDE.Models.FNO import *
from Neural_PDE.Utils.processing_utils import * 
from Neural_PDE.Utils.training_utils import * 

from Utils.plot_tools import subplots_2d

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
# Generating calibration data
n_cal = configuration['n_cal']
from pyDOE import lhs
from Neural_PDE.Numerical_Solvers.Burgers.Burgers_1D import * 

Nx = 1000 #Number of x-points
Nt = 500 #Number of time instances 
x_min = 0.0 #Min of X-range 
x_max = 2.0 #Max of X-range 
t_end = 1.25 #Time Maximum
nu = 0.002
x_slice = 5
t_slice = 10

# x_slice, t_slice = 1, 1
# alpha, beta, gamma = 1.0, 1.0, 1.0

lb = np.asarray([3, 3, 3]) # Lower Bound of the parameter domain
ub = np.asarray([5, 5, 5]) # Upper bound of the parameter domain

params = lb + (ub - lb) * lhs(3, n_cal)
u_sol = []
for ii in tqdm(range(n_cal)):
    sim = Burgers_1D(Nx, Nt, x_min, x_max, t_end, nu) 
    sim.InitializeU(params[ii,0], params[ii,1], params[ii,2])
    u_soln, x, dt = sim.solve()
    u_sol.append(u_soln)

u_sol = np.asarray(u_sol)[:, ::t_slice, ::x_slice]
x = x[::x_slice]
dt = dt*t_slice

u = torch.tensor(u_sol, dtype=torch.float32)
u = u.permute(0, 2, 1) #Adding BS and Permuting for FNO
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

from Utils.plot_tools import subplots_1d
x = x
values = {"Numerical": u_out[0,0].T, 
          "Prediction": pred[0,0].T
          }
indices = [6, 12, 18, 24]
subplots_1d(x, values, indices, "Comparing Model Prediction")
# %% 
#Estimating the Residuals of the calibration
uu = pred #Prediction
# uu = u_out #Solution

uu = uu.permute(0,1,3,2)
uu = uu[:, 0]
uu_cal = uu

dx = x[-1] - x[-2]
dt = dt

alpha = 1/dt*2
beta = 1/dx*2
gamma = 1/dx**2     

from Utils.ConvOps_1d import ConvOperator
#Defining the required Convolutional Operations. 
D_t = ConvOperator(domain='t', order=1)#, scale=alpha)
D_x = ConvOperator(domain='x', order=1)#, scale=beta) 
D_xx = ConvOperator(domain='x', order=2)#, scale=gamma)

# Residual
cal_residual  = dx*D_t(uu) + dt * uu * D_x(uu) - nu * D_xx(uu) * (2*dt/dx)

#Plotting
idx = 0
x_values = x
y_values = {"Prediction": cal_residual[idx][1:-1]
          }
indices = [6, 12, 18, 24]
subplots_1d(x_values, y_values, indices, "Residuals")

# %% 
#Generating Prediction Data
n_pred = configuration['n_pred']
lb = np.asarray([3, 3, 3]) # Lower Bound of the parameter domain
ub = np.asarray([5, 5, 5]) # Upper bound of the parameter domain

params = lb + (ub - lb) * lhs(3, n_pred)
u_sol = []
for ii in tqdm(range(n_pred)):
    sim = Burgers_1D(Nx, Nt, x_min, x_max, t_end, nu) 
    sim.InitializeU(params[ii,0], params[ii,1], params[ii,2])
    u_soln, x, dt = sim.solve()
    u_sol.append(u_soln)

u_sol = np.asarray(u_sol)[:, ::t_slice, ::x_slice]
x = x[::x_slice]
dt = dt*t_slice

u = torch.tensor(u_sol, dtype=torch.float32)
u = u.permute(0, 2, 1) #Adding BS and Permuting for FNO
u = u.unsqueeze(1) #Adding the variable channel

u_in = u[...,:configuration['T_in']]
u_out = u[...,configuration['T_in'] : configuration['T_in'] + configuration['T_out']]

pred, mse, mae = validation_AR(model, u_in, u_out, configuration['Step'], configuration['T_out'])
 # %% 
 #Estimating the residuals for the Prediction

uu = pred #Prediction
# uu = u_out #Solution

uu = uu.permute(0,1,3,2)
uu = uu[:, 0]
uu_pred = uu

dx = x[-1] - x[-2]
dt = dt

alpha = 1/dt*2
beta = 1/dx*2
gamma = 1/dx**2     

#Defining the required Convolutional Operations. 
D_t = ConvOperator(domain='t', order=1)#, scale=alpha)
D_x = ConvOperator(domain='x', order=1)#, scale=beta) 
D_xx = ConvOperator(domain='x', order=2)#, scale=gamma)

# Residual
pred_residual  = dx*D_t(uu) + dt * uu * D_x(uu) - nu * D_xx(uu) * (2*dt/dx)

# %%
#Performing CP over the residual space
from Neural_PDE.UQ.inductive_cp import * 
ncf_scores = np.abs(cal_residual.numpy()) #3/ (uu_cal.numpy() + 1e-6)
alpha = 0.1
qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)

##Input Independent
prediction_sets = [pred_residual.numpy() - qhat, pred_residual.numpy() + qhat]
##Input Dependent
# prediction_sets = [pred_residual.numpy() - qhat*uu_pred.numpy(), pred_residual.numpy() + qhat*uu_pred.numpy()]

# %% 

x_values = x[1:-1]
idx = 0
values = {"Residual": pred_residual[idx][1:-1, 1:-1], 
          "Lower": prediction_sets[0][idx][1:-1, 1:-1],
          "Upper": prediction_sets[1][idx][1:-1, 1:-1]
          }

indices = [6, 12, 18, 24]
subplots_1d(x_values, values, indices, "CP within the residual space.")

# %% 
#Checking for coverage:
#Obtaining the residuals for the Numerical Solution. 
uu = u_out
uu = uu.permute(0,1,3,2)
uu = uu[:,0]
val_residual = dx*D_t(uu) + dt * uu * D_x(uu) - nu * D_xx(uu) * (2*dt/dx)

#Emprical Coverage for all values of alpha 
alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov_res = []
for alpha in tqdm(alpha_levels):
    qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
    prediction_sets = [pred_residual.numpy() - qhat, pred_residual.numpy() + qhat]
    emp_cov_res.append(emp_cov(prediction_sets, val_residual.numpy()))

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()

# %%
