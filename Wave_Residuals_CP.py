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
from Neural_PDE.Models.FNO import *
from Neural_PDE.Utils.processing_utils import * 
from Neural_PDE.Utils.training_utils import * 

from plot_tools import subplots_2d

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
n_cal = 10 #Calibration dataset
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

dx = np.asarray(x[-1] - x[-2])
dy = np.asarray(y[-1] - y[-2])
dt = t[-1] - t[-2]

alpha = 1/dx**2
beta = 1/dt**2

from ConvOps import ConvOperator
#Defining the required Convolutional Operations. 
D_tt = ConvOperator('t', 2)#, scale=alpha)
D_xx_yy = ConvOperator(('x','y'), 2)#, scale=beta)

# %% Additive Kernels 
D = ConvOperator()
D.kernel = D_tt.kernel - (c*dt/dx)**2 * D_xx_yy.kernel 
u_residual = D(uu)[:, 1:-1,1:-1,1:-1] #Need to fix this padding. 


# %%
#Plotting the fields, prediction, abs error and the residual
idx = 0
t_idx = 10
values = [u_out[0, 0,...,t_idx][1:-1,1:-1],
          pred[0,0,...,t_idx][1:-1,1:-1],
          pred[0,0,...,t_idx][1:-1,1:-1] - u_out[0,0,...,t_idx][1:-1,1:-1],
          u_residual[idx, t_idx]
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
          pred[0,0,...,t_idx[0]][1:-1,1:-1] - u_out[0,0,...,t_idx[0]][1:-1,1:-1],
          u_residual[idx, t_idx[0]][1:-1,1:-1],
          pred[0,0,...,t_idx[1]][1:-1,1:-1] - u_out[0,0,...,t_idx[1]][1:-1,1:-1],
          u_residual[idx, t_idx[1]][1:-1,1:-1],
          pred[0,0,...,t_idx[2]][1:-1,1:-1] - u_out[0,0,...,t_idx[2]][1:-1,1:-1],
          u_residual[idx, t_idx[2]][1:-1,1:-1],
          ]

titles = ['Err: t=' + str(t_idx[0]),
          'Res: t=' + str(t_idx[0]),
          'Err: t=' + str(t_idx[1]),
          'Res: t=' + str(t_idx[1]),
          'Err: t=' + str(t_idx[2]),
          'Res: t=' + str(t_idx[2]),
          ]

subplots_2d(values, titles)

# %% 
#Performing CP over the residual space
from Neural_PDE.UQ.inductive_cp import * 
ncf_scores = u_residual
alpha = 0.1
qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
prediction_sets = [np.zeros(u_residual.shape) - qhat, np.zeros(u_residual.shape) + qhat]
empirical_coverage = emp_cov(prediction_sets, np.zeros(u_residual.shape))
print(f"The empirical coverage after calibration is: {empirical_coverage}")

#Checking for Coverage across various alpha levels 
alpha_levels = np.arange(0.05, 0.95, 0.1)
empirical_coverage = []
for ii in tqdm(range(len(alpha_levels))):
    qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha_levels[ii])
    prediction_sets = [np.zeros(u_residual.shape) - qhat, np.zeros(u_residual.shape) + qhat]
    empirical_coverage.append(emp_cov(prediction_sets, np.zeros(u_residual.shape)))

#Plotting Coverage
import matplotlib as mpl
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.75)
plt.plot(1-alpha_levels, empirical_coverage, label='Residual' ,ls='-.', color='teal', alpha=0.75)
plt.xlabel(r'1-$\alpha$')
plt.ylabel('Empirical Coverage')
# %%
