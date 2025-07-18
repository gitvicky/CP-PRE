#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uncertainty quantification of a Neural PDE Surrogate solving the Wave Equation using Physics Residuals and guaranteed using Conformal Prediction

Equation : u_tt = c**2 * (u_xx + u_yy)
Surrogate Model : FNO
Numerical Solver : https://github.com/gitvicky/Neural_PDE/blob/main/Numerical_Solvers/Wave/Wave_2D_Spectral.py

"""

# %%
configuration = {"Case": 'Wave',
                 "Field": 'u',
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
                 "T_in": 20,    
                 "T_out": 60,
                 "Step": 10,
                 "Width_time": 32, 
                 "Width_vars": 0,  
                 "Modes": 8,
                 "Variables":1, 
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
from pyDOE import lhs

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
# Generating the calibration dataset through simulations
n_cal = configuration['n_cal'] #Calibration dataset

Nx = 64 # Mesh Discretesiation 
x_min = -1.0 # Minimum value of x
x_max = 1.0 # maximum value of x
y_min = -1.0 # Minimum value of y 
y_max = 1.0 # Minimum value of y
tend = 1
c = 1.0 #Wave Speed <=1.0
    
#Initial Condition Parameterisations: amplitude, x and y positions of the initial gaussian. 
lb = np.asarray([10, 0.10, 0.10]) #amp, xx_pos, y_pos
ub = np.asarray([50, 0.50, 0.50]) 
params = lb + (ub-lb)*lhs(3, n_cal)

#Initialising the Solver
from Neural_PDE.Numerical_Solvers.Wave.Wave_2D_Spectral import * 
u_sol = []
solver = Wave_2D(Nx, x_min, x_max, tend, c)
for ii in tqdm(range(n_cal)):
    x, y, t, uu = solver.solve(params[ii,0], params[ii,1], params[ii,2])
    u_sol.append(uu[::5])
    t = t[::5]
u_sol = np.asarray(u_sol)

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

u = torch.tensor(u_sol, dtype=torch.float32)#converting to torch 
u = u.permute(0, 2, 3, 1)#BS, Nx, Ny, Ntx_min = -1.0 # Minimum value of x
x_max = 1.0 # maximum value of x
y_min = -1.0 # Minimum value of y 
y_max = 1.0 # Minimum value of y
tend = 1
u = u.unsqueeze(1)#BS, vars, Nx, Ny, Nt

u_in = u[...,:configuration['T_in']]
u_out = u[...,configuration['T_in'] : configuration['T_in'] + configuration['T_out']]

# %% 
#Normalisation
norms = np.load(model_loc + '/FNO_Wave_charitable-sea_norms.npz')
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
#Load the model. 
model = FNO_multi2d(configuration['T_in'], configuration['Step'], configuration['Modes'], configuration['Modes'], configuration['Variables'], configuration['Width_time'])
model.load_state_dict(torch.load(model_loc + '/FNO_Wave_charitable-sea.pth', map_location='cpu'))

#Model Predictions.
pred_encoded, mse, mae = validation_AR(model, u_in, u_out_encoded, configuration['Step'], configuration['T_out'])

print('(MSE) Error: %.3e' % (mse))
print('(MAE) Error: %.3e' % (mae))

#Denormalising the predictions
pred = out_normalizer.decode(pred_encoded.to(device)).cpu()

# %% 
#Estimating the Residuals ->  u_tt  = c**2 * (u_xx + u_yy)

# uu = u_out[:, 0] #Validating on Numerical Solution 
uu = pred[:, 0] #Prediction
uu = uu.permute(0, 3, 1, 2) #BS, Nt, Nx, Ny
uu_cal = uu
dx = np.asarray(x[-1] - x[-2])
dy = np.asarray(y[-1] - y[-2])
dt = t[-1] - t[-2]

alpha = 1/dx**2
beta = 1/dt**2

from Utils.ConvOps_2d import ConvOperator
#Defining the required Convolutional Operations. 
D_tt = ConvOperator('t', 2)#, scale=alpha)
D_xx_yy = ConvOperator(('x','y'), 2)#, scale=beta)

# %% Additive Kernels 
D = ConvOperator()
D.kernel = D_tt.kernel - (c*dt/dx)**2 * D_xx_yy.kernel 
cal_residual = D(uu)#[:, 1:-1,1:-1,1:-1] #Need to fix this padding. 

# %%
#Plotting the fields, prediction, abs error and the residual
idx = 0
t_idx = 10
values = [u_out[0, 0,...,t_idx],
          pred[0,0,...,t_idx],
          pred[0,0,...,t_idx] - u_out[0,0,...,t_idx],
          cal_residual[idx, t_idx]
          ]

titles = [r'$u$',
          r'$\tilde u$',
          r'$u - \tilde u$',
          r'$ \frac{\partial^2 \tilde u}{\partial t^2 } - c^2 (\frac{\partial^2 \tilde u}{\partial x^2} + \frac{\partial^2 \tilde u}{\partial y^2})$'
          ]

subplots_2d(values, titles)

# %% 
# Plotting the Error and the Residuals across time
idx = 0
t_idx = [10, 20, 30]
values = [
          pred[0,0,...,t_idx[0]] - u_out[0,0,...,t_idx[0]],
          cal_residual[idx, t_idx[0]],
          pred[0,0,...,t_idx[1]] - u_out[0,0,...,t_idx[1]],
          cal_residual[idx, t_idx[1]],
          pred[0,0,...,t_idx[2]] - u_out[0,0,...,t_idx[2]],
          cal_residual[idx, t_idx[2]],
          ]

titles = ['Err: t=' + str(t_idx[0]),
          'Res: t=' + str(t_idx[0]),
          'Err: t=' + str(t_idx[1]),
          'Res: t=' + str(t_idx[1]),
          'Err: t=' + str(t_idx[2]),
          'Res: t=' + str(t_idx[2]),
          ]

subplots_2d(values, titles, )

# %% 
#Generating Predictions 
n_pred = configuration['n_pred'] #Prediction Dataset

#Initial Condition Parameterisations: amplitude, x and y positions of the initial gaussian. 
# lb = np.asarray([20, -0.50, -0.50]) #amp, xx_pos, y_pos
# ub = np.asarray([40, 0.50, 0.50]) 
params = lb + (ub-lb)*lhs(3, n_pred)

u_sol = []
solver = Wave_2D(Nx, x_min, x_max, tend, c)
for ii in tqdm(range(n_pred)):
    x, y, t, uu = solver.solve(params[ii,0], params[ii,1], params[ii,2])
    u_sol.append(uu[::5])
    t = t[::5]
u_sol = np.asarray(u_sol)

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

u = torch.tensor(u_sol, dtype=torch.float32)#converting to torch 
u = u.permute(0, 2, 3, 1)#BS, Nx, Ny, Ntx_min = -1.0 # Minimum value of x
u = u.unsqueeze(1)#BS, vars, Nx, Ny, Nt

u_in = u[...,:configuration['T_in']]
u_out = u[...,configuration['T_in'] : configuration['T_in'] + configuration['T_out']]

#Normalisation
u_in = in_normalizer.encode(u_in)
u_out_encoded = out_normalizer.encode(u_out)

#Model Predictions.
pred_encoded, mse, mae = validation_AR(model, u_in, u_out_encoded, configuration['Step'], configuration['T_out'])

print('(MSE) Error: %.3e' % (mse))
print('(MAE) Error: %.3e' % (mae))

#Denormalising the predictions
pred = out_normalizer.decode(pred_encoded.to(device)).cpu()

# %% 
#Obtaining the residuals for the predictions.
uu = pred[:,0]
uu = uu.permute(0, 3, 1, 2) #BS, Nt, Nx, Ny
uu_pred = uu
pred_residual = D(uu)
# %% 
#Performing CP over the residual space
from Neural_PDE.UQ.inductive_cp import * 
ncf_scores = np.abs(cal_residual.numpy())# / (uu_cal.numpy() + 1e-6) #MINE
# ncf_scores = np.abs(cal_residual.numpy()) / (np.std(uu_cal.numpy(), axis= 0) + 1e-6) #NICP
alpha = 0.1
qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
prediction_sets = [pred_residual.numpy() - qhat, pred_residual.numpy() + qhat] 
#Input Dependent- MINE 
# prediction_sets = [pred_residual.numpy() - qhat*uu_pred.numpy(), pred_residual.numpy() + qhat*uu_pred.numpy()]
# #NICP 
# prediction_sets = [pred_residual.numpy() - qhat*np.std(uu_pred.numpy(), axis=0), pred_residual.numpy() + qhat*np.std(uu_pred.numpy(), axis=0)]

# %%
# Plotting the Residual and the qhat 
idx = 9
t_idx = 20
values = [
          uu_pred[idx,t_idx],
          pred_residual[idx,t_idx],
          prediction_sets[0][idx, t_idx],
          prediction_sets[1][idx, t_idx]
          ]

titles = ['Prediction: t=' + str(t_idx),
          'Residual: t=' + str(t_idx),
          'Lower Bar: t=' + str(t_idx),
          'Upper Bar: t=' + str(t_idx)
          ]

subplots_2d(values, titles, "Conformal Prediction")

# %% 
# #Checking for coverage:
#Obtaining the residuals for the Numerical Solution. 
uu = u_out[:,0]
uu = uu.permute(0, 3, 1, 2) #BS, Nt, Nx, Ny
val_residual = D(uu)

#Emprical Coverage for all values of alpha 
alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov_res = []
for alpha in tqdm(alpha_levels):
    qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
    prediction_sets = [pred_residual.numpy() - qhat, pred_residual.numpy() + qhat]
    # prediction_sets = [pred_residual.numpy() - qhat*uu_pred.numpy(), pred_residual.numpy() + qhat*uu_pred.numpy()] #MINE
    emp_cov_res.append(emp_cov(prediction_sets, val_residual.numpy()))

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()
# %%
#Inverting the Bounds 
u_lower = D.integrate(torch.tensor(prediction_sets[0]))
u_upper = D.integrate(torch.tensor(prediction_sets[1]))

 # Plotting the Residual and the qhat 
idx = 0
t_idx = 10
values = [
          u_out[idx, 0, :, :, t_idx],
          uu_pred[idx, t_idx],
          u_lower[idx, t_idx],
          u_upper[idx, t_idx]
          ]

titles = ['Solution: t=' + str(t_idx),
          'Prediction: t=' + str(t_idx),
          'Lower Bar: t=' + str(t_idx),
          'Upper Bar: t=' + str(t_idx)
          ]

# subplots_2d(values, titles, "Inverted Bounds")

# %%
