#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uncertainty quantification of a Neural PDE Surrogate solving the Advection Equation using Physics Residuals and guaranteed using Conformal Prediction

Equation :  U_t + v U_x = 0
Surrogate Model : FNO
Numerical Solver : https://github.com/gitvicky/Neural_PDE/blob/main/Numerical_Solvers/Advection/Advection_1D.py
"""

# %% Training Configuration - used as the config file for simvue.
configuration = {"Case": 'Advection',
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
                 "n_train": 100,
                 "n_cal": 100,
                 "n_pred": 100
                 }

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
#Setting up locations. 
file_loc = os.getcwd()

#Setting up the seeds and devices
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

# %% 
#Generating the Datasets by running the simulation
t1 = default_timer()
from Neural_PDE.Numerical_Solvers.Advection.Advection_1D import *
from pyDOE import lhs

#Obtaining the exact and FD solution of the 1D Advection Equation. 

Nx = 200 #Number of x-points
Nt = 50 #Number of time instances 
x_min, x_max = 0.0, 2.0 #X min and max
t_end = 0.5 #time length

sim = Advection_1d(Nx, Nt, x_min, x_max, t_end) 

n_train = configuration['n_train']

lb = np.asarray([0.1, 0.1]) #pos, velocity
ub = np.asarray([1.0, 1.0])

params = lb + (ub - lb) * lhs(2, n_train)

u_sol = []
for ii in tqdm(range(n_train)):
    xc = params[ii, 0]
    v = params[ii, 1]
    x, t, u_soln, u_exact = sim.solve(v, xc)
    u_sol.append(u_soln)

u_sol = np.asarray(u_sol)
u_sol = u_sol[:, :, 1:-2]
x = x[1:-2]
velocity = params[:,1]

# %% 
u = torch.tensor(u_sol, dtype=torch.float32)
u = u.permute(0, 2, 1) #only for FNO
u = u.unsqueeze(1)
x_grid = x
t_grid = t
# %% 
ntrain = int(0.8 * n_train)
ntest = int(0.2 * n_train)
S = Nx  #Grid Size

#Extracting configuration files
T_in = configuration['T_in']
T_out = configuration['T_out']
step = configuration['Step']
width = configuration['Width']
modes = configuration['Modes']
output_size = configuration['Step']
num_vars = configuration['Variables']
batch_size = configuration['Batch Size']

train_a = u[:ntrain,:, :, :T_in]
train_u = u[:ntrain,:, :, T_in:T_out+T_in]

test_a = u[-ntest:, :, :, :T_in]
test_u = u[-ntest:, :, :, T_in:T_out+T_in]

print("Training Input: " + str(train_a.shape))
print("Training Output: " + str(train_u.shape))

#No Normalisation -- Normalisation = Identity 

# %%
#Setting up the training and testing data splits
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %%
################################################################
# training and evaluation
################################################################
model = FNO_multi1d(T_in, step, modes, num_vars, width, width_vars=0)
model.to(device)

print("Number of model params : " + str(model.count_params()))

#Setting up the optimizer and scheduler, loss and epochs 
optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
loss_func = torch.nn.MSELoss()
epochs = configuration['Epochs']

# %%
####################################
#Training Loop 
####################################
start_time = default_timer()
for ep in range(epochs): #Training Loop - Epochwise

    model.train()
    t1 = default_timer()
    train_loss, test_loss = train_one_epoch(model, train_loader, test_loader, loss_func, optimizer)
    t2 = default_timer()

    train_loss = train_loss / ntrain / num_vars
    test_loss = test_loss / ntest / num_vars

    print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 3)}, Test Loss: {round(test_loss,3)}")
    
    scheduler.step()

train_time = default_timer() - start_time

# %%
#Inference
pred_set, mse, mae = validation_AR(model, test_a, test_u, step, T_out)

print('Testing Error (MSE) : %.3e' % (mse))
print('Testing Error (MAE) : %.3e' % (mae))


# %% 
# Performing the calibration. 

n_cal = configuration['n_cal']

lb = np.asarray([0.1, 0.1]) #pos, velocity
ub = np.asarray([1.0, 1.0])

params = lb + (ub - lb) * lhs(2, n_train)

u_sol = []
for ii in tqdm(range(n_cal)):
    xc = params[ii, 0]
    v = params[ii, 1]
    x, t, u_soln, u_exact = sim.solve(v, xc)
    u_sol.append(u_soln)

u_sol = np.asarray(u_sol)
u_sol = u_sol[:, :, 1:-2]
x = x[1:-2]
velocity = params[:,1]

u = torch.tensor(u_sol, dtype=torch.float32)
u = u.permute(0, 2, 1) #only for FNO
u = u.unsqueeze(1)

u_in = u[...,:configuration['T_in']]
u_out = u[...,configuration['T_in'] : configuration['T_in'] + configuration['T_out']]

# %%
#Model Predictions.
pred, mse, mae = validation_AR(model, u_in, u_out, configuration['Step'], configuration['T_out'])

# %% 
#Estimating the Residuals

uu = pred #Prediction
# uu = u_out #Solution

uu = uu.permute(0,1,3,2)
uu = uu[:,0]
uu_cal = uu

dx = x[-1] - x[-2]
dt = t[-1] - t[-2]

alpha = 1/dt*2
beta = 1/dx*2

from ConvOps_1d import ConvOperator
#Defining the required Convolutional Operations. 
D_t = ConvOperator(domain='t', order=1)#, scale=alpha)
D_x = ConvOperator(domain='x', order=1)#, scale=beta) 

#Residual - Additive Kernels
D = ConvOperator()
D.kernel = D_t.kernel + (v*dt/dx) * D_x.kernel
cal_residual = D(uu_cal)

# %% 
from plot_tools import subplots_2d
values = [cal_residual[0, 1:-1, 1:-1]]
titles = ["Residual"]

subplots_2d(values, titles)
# %%
#Generating Prediction Data

n_pred = configuration['n_pred']

lb = np.asarray([0.1, 0.1]) #pos, velocity
ub = np.asarray([1.0, 1.0])

params = lb + (ub - lb) * lhs(2, n_train)

u_sol = []
for ii in tqdm(range(n_pred)):
    xc = params[ii, 0]
    v = params[ii, 1]
    x, t, u_soln, u_exact = sim.solve(v, xc)
    u_sol.append(u_soln)

u_sol = np.asarray(u_sol)
u_sol = u_sol[:, :, 1:-2]
x = x[1:-2]
velocity = params[:,1]

u = torch.tensor(u_sol, dtype=torch.float32)
u = u.permute(0, 2, 1) #only for FNO
u = u.unsqueeze(1)

u_in = u[...,:configuration['T_in']]
u_out = u[...,configuration['T_in'] : configuration['T_in'] + configuration['T_out']]

pred, mse, mae = validation_AR(model, u_in, u_out, configuration['Step'], configuration['T_out'])

# %% 
#Estimating the Residuals
uu = pred #Prediction
# uu = u_out #Solution

uu = uu.permute(0,1,3,2)
uu = uu[:,0]
uu_pred = uu

pred_residual = D(uu_pred)

# %%
#Performing CP over the residual space
from Neural_PDE.UQ.inductive_cp import * 
ncf_scores = np.abs(cal_residual.numpy()) #3/ (uu_cal.numpy() + 1e-6)
alpha = 0.5
qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)

##Input Independent
prediction_sets = [pred_residual.numpy() - qhat, pred_residual.numpy() + qhat]
##Input Dependent
# prediction_sets = [pred_residual.numpy() - qhat*uu_pred.numpy(), pred_residual.numpy() + qhat*uu_pred.numpy()]

# %% 

from plot_tools import subplots_1d
x_values = x[1:-1]
idx = 5
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
val_residual = D(uu)

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
#Inverting the Bounds 
u_lower = D.integrate(torch.tensor(prediction_sets[0]))
u_upper = D.integrate(torch.tensor(prediction_sets[1]))

x_values = x[1:-1]
idx = 5
values = {"Residual":uu[idx][1:-1, 1:-1], 
          "Lower": u_lower[idx][1:-1, 1:-1],
          "Upper": u_upper[idx][1:-1, 1:-1]
          }

indices = [6, 12, 18, 24]
subplots_1d(x_values, values, indices, "CP within the residual space.")

# %%
