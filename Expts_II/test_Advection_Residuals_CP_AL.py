#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uncertainty quantification of a Neural PDE Surrogate solving the Advection Equation using Physics Residuals and guaranteed using Conformal Prediction
Using the Residual Information and the obtained error bars for AL. 

Equation :  U_t + v U_x = 0
Surrogate Model : FNO
Numerical Solver : https://github.com/gitvicky/Neural_PDE/blob/main/Numerical_Solvers/Advection/Advection_1D.py
"""

# %% Training Configuration - used as the config file for simvue.
configuration = {"Case": 'Advection',
                 "Field": 'u',
                 "Model": 'FNO',
                 "Epochs": 100,
                 "Batch Size": 10,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.001,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'Tanh',
                 "Normalisation Strategy": 'Identity',
                 "T_in": 1,    
                 "T_out": 10,
                 "Step": 1,
                 "Width": 16, 
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
import sys
sys.path.append("..")
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
#Simulation Setup
t1 = default_timer()
from Neural_PDE.Numerical_Solvers.Advection.Advection_1D import *
from pyDOE import lhs

#Obtaining the exact and FD solution of the 1D Advection Equation. 
Nx = 200 #Number of x-points
Nt = 50 #Number of time instances 
x_min, x_max = 0.0, 2.0 #X min and max
t_end = 0.5 #time length
v = 1.0
sim = Advection_1d(Nx, Nt, x_min, x_max, t_end) 
n_train = configuration['n_train']
# %% 
#Define Bounds
lb = np.asarray([0.1, 50]) #pos, amplitude
ub = np.asarray([1.0, 200])

params = lb + (ub - lb) * lhs(2, n_train)

#Generating Data 
u_sol = []
for ii in tqdm(range(n_train)):
    xc = params[ii, 0]
    amp = params[ii, 1]
    x, t, u_soln, u_exact = sim.solve(xc, amp, v)
    u_sol.append(u_soln)

#Extraction
u_sol = np.asarray(u_sol)
u_sol = u_sol[:, :, 1:-2]
x = x[1:-2]
velocity = params[:,1]

#Tensorize
u = torch.tensor(u_sol, dtype=torch.float32)
u = u.permute(0, 2, 1) #only for FNO
u = u.unsqueeze(1)
x_grid = x
t_grid = t
# %% 
#Train-Test
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
    train_loss, test_loss = train_one_epoch_AR(model, train_loader, test_loader, loss_func, optimizer, step, T_out)
    t2 = default_timer()

    train_loss = train_loss / ntrain / num_vars
    test_loss = test_loss / ntest / num_vars

    print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 3)}, Test Loss: {round(test_loss,3)}")
    
    scheduler.step()

train_time = default_timer() - start_time

# %%
#Evaluation
pred_set, mse, mae = validation_AR(model, test_a, test_u, step, T_out)

print('Testing Error (MSE) : %.3e' % (mse))
print('Testing Error (MAE) : %.3e' % (mae))


# %% 
# Calibrating over new samples within the defined domain using the PDE residuals. 

n_cal = configuration['n_cal']
params = lb + (ub - lb) * lhs(2, n_cal)

u_ic_cal = []
for ii in tqdm(range(n_cal)):
    xc = params[ii, 0]
    amp = params[ii, 1]
    sim.initializeU(xc, amp)
    u_ic_cal.append(sim.u)

u_ic_cal = np.asarray(u_ic_cal)[:, 1:-2]

u_in_cal = torch.tensor(u_ic_cal, dtype=torch.float32).unsqueeze(1).unsqueeze(-1)
u_out_cal_shape = [n_cal, 1, len(x), T_out]
# %%
#Model Predictions.
pred_cal, mse, mae = validation_AR(model, u_in_cal, torch.zeros(u_out_cal_shape), configuration['Step'], configuration['T_out'])

# %% 
#Estimating the Residuals

uu = pred_cal #Prediction
uu = uu.permute(0,1,3,2)
uu = uu[:,0]
uu_cal = uu

dx = x[-1] - x[-2]
dt = t[-1] - t[-2]

alpha = 1/dt*2
beta = 1/dx*2

from Utils.ConvOps_1d import ConvOperator
#Defining the required Convolutional Operations. 
D_t = ConvOperator(domain='t', order=1)#, scale=alpha)
D_x = ConvOperator(domain='x', order=1)#, scale=beta) 

#Residual - Additive Kernels
D = ConvOperator()
D.kernel = D_t.kernel + (v*dt/dx) * D_x.kernel
cal_residual = D(uu_cal)

#Plotting
from Utils.plot_tools import subplots_2d
values = [cal_residual[0, 1:-1, 1:-1]]
titles = ["Residual"]

subplots_2d(values, titles)

#Performing CP over the residual space
from Neural_PDE.UQ.inductive_cp import * 
ncf_scores = np.abs(cal_residual.numpy()) 
# alpha = 0.5
# qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)

# %%
#Generating Prediction Data

n_pred = configuration['n_pred']
params = lb + (ub - lb) * lhs(2, n_pred)

u_ic_pred = []
for ii in tqdm(range(n_cal)):
    xc = params[ii, 0]
    amp = params[ii, 1]
    sim.initializeU(xc, amp)
    u_ic_pred.append(sim.u)

u_ic_pred = np.asarray(u_ic_pred)[:, 1:-2]

u_in_pred = torch.tensor(u_ic_pred, dtype=torch.float32).unsqueeze(1).unsqueeze(-1)
u_out_pred_shape = [n_pred, 1, len(x), T_out]

#Model Predictions.
pred_pred, mse, mae = validation_AR(model, u_in_pred, torch.zeros(u_out_pred_shape), configuration['Step'], configuration['T_out'])

#Estimating the Residuals of the prediction
uu = pred_pred

uu = uu.permute(0,1,3,2)
uu = uu[:,0]
uu_pred = uu

pred_residual = D(uu_pred)

# %% 
#Sim Selection / Rejection if % cells outside of bounds > threshold

def filter_sims_within_bounds(lower_bound, upper_bound, samples, threshold, within=False):
    """
    Filter samples that have values within the given bounds at least threshold percent of the time.
    
    Parameters:
    lower_bound (np.array): Lower bound array of shape [Nt, Nx]
    upper_bound (np.array): Upper bound array of shape [Nt, Nx]
    samples (np.array): Sample array of shape [BS, Nt, Nx]
    threshold (float): Minimum percentage of values that must be within bounds
    within (boolean): values within or outside the bounds

    Returns:
    np.array: Boolean array indicating which samples meet the criterion
    """
    # Ensure inputs are numpy arrays
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)
    samples = np.array(samples)
    
    # Check if samples are within bounds
    if within:
        with_bounds = (samples >= lower_bound) & (samples <= upper_bound)
    else: 
        with_bounds = (samples <= lower_bound) | (samples >= upper_bound)

    # Calculate the percentage of values in/out bounds for each sample
    percent_with_bounds = with_bounds.mean(axis=(1,2))
    
    # Return boolean array indicating which samples meet the threshold
    return percent_with_bounds >= threshold

alpha = 0.5
qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
filtered_sims = filter_sims_within_bounds(-qhat, qhat, pred_residual.numpy(), threshold=0.5)

# %%
#Generate further training samples based on the rejected ones
params_filtered = params[filtered_sims]
#Generating Data 
u_sol = []
for ii in tqdm(range(len(params_filtered))):
    xc = params_filtered[ii, 0]
    amp = params_filtered[ii, 1]
    x, t, u_soln, u_exact = sim.solve(xc, amp, v)
    u_sol.append(u_soln)

#Extraction
u_sol = np.asarray(u_sol)
u_sol = u_sol[:, :, 1:-2]
x = x[1:-2]
velocity = 1 

#Tensorize
u = torch.tensor(u_sol, dtype=torch.float32)
u = u.permute(0, 2, 1) #only for FNO
u = u.unsqueeze(1)
x_grid = x
t_grid = t
# %% 
#Train-Test
S = Nx  #Grid Size

train_a = u[:, :, :, :T_in]
train_u = u[:, :, :, T_in:T_out+T_in]

print("Training Input: " + str(train_a.shape))
print("Training Output: " + str(train_u.shape))

#No Normalisation -- Normalisation = Identity 

#Setting up the training and testing data splits
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

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
    train_loss, test_loss = train_one_epoch_AR(model, train_loader, test_loader, loss_func, optimizer, step, T_out)
    t2 = default_timer()

    train_loss = train_loss / ntrain / num_vars
    test_loss = test_loss / ntest / num_vars

    print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 3)}, Test Loss: {round(test_loss,3)}")
    
    scheduler.step()

train_time = default_timer() - start_time

#Evaluation
pred_set, mse, mae = validation_AR(model, test_a, test_u, step, T_out)

print('Testing Error (MSE) : %.3e' % (mse))
print('Testing Error (MAE) : %.3e' % (mae))

# %% 