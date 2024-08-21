#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uncertainty quantification of a Neural PDE Surrogate solving the Advection Equation using Physics Residuals and guaranteed using Conformal Prediction
but we are testing to see if we can sample a million forward iterations, see which ones fit within the CP bars and then using that samples to form the envelopes 
that correspond to the 

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
                 "n_cal": 1000,
                 "n_pred": 10000
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

lb = np.asarray([0.5, 50]) #pos, amplitude
ub = np.asarray([1.0, 200])

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
#Evaluation
pred_set, mse, mae = validation_AR(model, test_a, test_u, step, T_out)

print('Testing Error (MSE) : %.3e' % (mse))
print('Testing Error (MAE) : %.3e' % (mae))

# %% 
# Performing the calibration. 

n_cal = configuration['n_cal']

lb = np.asarray([0.5, 50]) #pos, amplitude
ub = np.asarray([1.0, 200])

params = lb + (ub - lb) * lhs(2, n_cal)

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

from Utils.ConvOps_1d import ConvOperator
#Defining the required Convolutional Operations. 
D_t = ConvOperator(domain='t', order=1)#, scale=alpha)
D_x = ConvOperator(domain='x', order=1)#, scale=beta) 

#Residual - Additive Kernels
D = ConvOperator()
D.kernel = D_t.kernel + (v*dt/dx) * D_x.kernel
cal_residual = D(uu_cal)

# %% 
#Performing CP over the residual space
from Neural_PDE.UQ.inductive_cp import * 
ncf_scores = np.abs(cal_residual.numpy()) #3/ (uu_cal.numpy() + 1e-6)
alpha = 0.1
qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)

##Input Independent
# prediction_sets = [pred_residual.numpy() - qhat, pred_residual.numpy() + qhat]
##Input Dependent
# prediction_sets = [pred_residual.numpy() - qhat*uu_pred.numpy(), pred_residual.numpy() + qhat*uu_pred.numpy()]

# %%
#Generating Prediction Data

n_pred = configuration['n_pred']

lb = np.asarray([0.5, 50]) #pos, amplitude
ub = np.asarray([1.0, 200])

params = lb + (ub - lb) * lhs(2, n_pred)

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
from Utils.plot_tools import subplots_1d

x_values = x
values = {
          "Lower": -qhat,
          "Upper": qhat,
          "pred_residual": pred_residual[5]
          }

indices = [6, 12, 18, 24]
subplots_1d(x_values, values, indices, "qhat")

# %% 
#Sampling to get the inverse error bar -- Vicky's version. 
sampling_residual_np = np.asarray(pred_residual)

import numpy as np

def filter_samples_within_bounds(lower_bound, upper_bound, samples, threshold=0.8):
    """
    Filter samples that have values within the given bounds at least threshold percent of the time.
    
    Parameters:
    lower_bound (np.array): Lower bound array of shape [30, 200]
    upper_bound (np.array): Upper bound array of shape [30, 200]
    samples (np.array): Sample array of shape [1000, 30, 200]
    threshold (float): Minimum percentage of values that must be within bounds (default: 0.8)
    
    Returns:
    np.array: Boolean array indicating which samples meet the criterion
    """
    # Ensure inputs are numpy arrays
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)
    samples = np.array(samples)
    
    # Check if samples are within bounds
    within_bounds = (samples >= lower_bound) & (samples <= upper_bound)
    
    # Calculate the percentage of values within bounds for each sample
    percent_within_bounds = within_bounds.mean(axis=(1,2))
    
    # Return boolean array indicating which samples meet the threshold
    return percent_within_bounds >= threshold

# Example usage:
# Assuming you have your arrays defined as lower_bound, upper_bound, and samples

alpha = 0.1
qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
filtered_samples = filter_samples_within_bounds(-qhat, qhat, sampling_residual_np, threshold=0.9)

# Get the indices of the samples that meet the criterion
selected_sample_indices = np.where(filtered_samples)[0]

# Print the number of samples that meet the criterion
print(f"Number of samples meeting the criterion: {selected_sample_indices.size}")

# If you want to get the actual filtered samples:
selected_samples = sampling_residual_np[filtered_samples]
uu_selected = uu_pred[filtered_samples]

# %% 


# ############################ Marginal - Ander's version #####################

# sampling_residual_np = np.asarray(sampling_residual[:-1])
# pred_residual_np = np.asarray(sampling_residual[1:2,:,:])
# # in_bounds = np.all(pred_residual_np <= qhat, axis = (1,2))
# # np.all(pred_residual_np <= qhat, pred_residual_np >= -qhat)

# (i1, i2) = qhat.shape

# # in_bounds = [[np.all(np.vstack([pred_residual_np[:,i,ii]-qhat[i, ii] <= sampling_residual_np[:,i,ii], sampling_residual_np[:,i,ii] <=  pred_residual_np[:,i,ii]+ qhat[i, ii]]).T, axis = 1) for i in range(i1)] for ii in range(i2)]

# in_bounds = [[np.all(np.vstack([-np.abs(pred_residual_np[:,i,ii]) <= sampling_residual_np[:,i,ii], sampling_residual_np[:,i,ii] <=  np.abs(pred_residual_np[:,i,ii])]).T, axis = 1) for i in range(i1)] for ii in range(i2)]

# in_bounds = np.asarray(in_bounds)

# in_bounds = in_bounds.reshape(uu_pred.shape[0]-1, uu_pred.shape[1], uu_pred.shape[2])
# Bounds_physical = np.zeros((2, i1, i2))

# uu_pred_np = np.asarray(uu_pred[:-1])

# for i in range(i1):
#     for ii in range(i2):
#         Bounds_physical[0, i, ii] = np.max(uu_pred_np[in_bounds[:, i, ii], i, ii])
#         Bounds_physical[1, i, ii] = np.min(uu_pred_np[in_bounds[:, i, ii], i, ii])


# # for ii in tqdm(len(pred_residual)):
# #     if pred_residual[ii]
# # %%
# from Utils.plot_tools import subplots_1d

# x_values = x
# values = {
#         #   "Sim": uu
#           "Lower": Bounds_physical[0],
#           "Upper": Bounds_physical[1]
#           }

# indices = [6, 12, 18, 24]
# subplots_1d(x_values, values, indices, "CP within the Physical space - hopefully.")

# # %%

# from Utils.plot_tools import subplots_1d

# x_values = x
# values = {
#         #   "Sim": uu
#         #   "Lower": Bounds_physical[0],
#         #   "Samps": sampling_residual_np[14],
#         "Samps": cal_residual[22], 
#         #   "Upper": np.abs(pred_residual_np[0]),
#         #   "Lower": -np.abs(pred_residual_np[0]),
#         #   "Upper": np.abs(pred_residual_np[0]),
#         #   "Lower": -np.abs(pred_residual_np[0]),
#           "qhat_lower" : -qhat,
#           "qhat_upper" : qhat
#           }

# indices = [6, 12, 18, 24]
# subplots_1d(x_values, values, indices, "CP within the Physical space - hopefully.")
# %% 
# # %%
# ################################Joint Coverage################################## 
pred_residual_np = pred_residual.numpy()
modulation = np.std(cal_residual[:, 1:-1,1:-1].numpy(), axis = 0)

def conf_metric_joint(residual):
    return np.max(np.abs(residual)/modulation,  axis = (1,2))

qhat = calibrate(scores=conf_metric_joint(cal_residual[:, 1:-1,1:-1].numpy()), n=len(cal_residual.numpy()), alpha=0.9)
prediction_sets =  [pred_residual_np[:, 1:-1,1:-1]-qhat*modulation, pred_residual_np[:, 1:-1,1:-1] + qhat*modulation]

x_values = x[1:-1]
values = {
          "Lower": prediction_sets[0][0],
          "Upper": prediction_sets[1][0]
          }

indices = [6, 12, 18, 24]
subplots_1d(x_values, values, indices, "CP within the Physical space - hopefully.")

# %% 
#Testing the impact of boundaries 
cal_res = cal_residual[:, 1:-1,1:-1]
modulation = np.std(cal_res.numpy(), axis = 0)
qhat = calibrate(scores=conf_metric_joint(cal_res.numpy()), n=len(cal_residual.numpy()), alpha=0.9)
print(qhat)
# %%
#Emprical Joint Coverage for all values of alpha 

uu = u_out
uu = uu.permute(0,1,3,2)
uu = uu[:,0]
val_residual = D(uu)

alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov_res = []
ncf_scores = conf_metric_joint(cal_residual[:, 1:-1, 1:-1].numpy())
for alpha in tqdm(alpha_levels):
    qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
    prediction_sets = [pred_residual[:,1:-1,1:-1].numpy() - qhat*modulation, pred_residual[:,1:-1,1:-1].numpy() + qhat*modulation]
    emp_cov_res.append(((val_residual[:,1:-1,1:-1].numpy() >= prediction_sets[0]).all(axis = (1,2)) & (val_residual[:,1:-1,1:-1].numpy() <= prediction_sets[1]).all(axis = (1,2))).mean())

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()

# %% 
# # %% 
# (i1, i2) = (30, 200)

# in_bounds_lower = prediction_sets[0][0] <= sampling_residual_np
# in_bounds_upper = prediction_sets[1][0] >= sampling_residual_np

# in_bounds_lower_all = np.all(in_bounds_lower, axis = (1,2))
# in_bounds_upper_all = np.all(in_bounds_lower, axis = (1,2))

# in_bounds_joint = np.logical_and(in_bounds_lower_all, in_bounds_upper_all)

# Bounds_physical = np.zeros((2, i1, i2))

# uu_pred_np = np.asarray(uu_pred[:-1])

# Bounds_physical[0] = np.min(uu_pred_np[in_bounds], axis = 0)
# Bounds_physical[1] = np.max(uu_pred_np[in_bounds], axis = 0)

# # %%

# x_values = x
# values = {
#           "Lower": Bounds_physical[0],
#           "Upper": Bounds_physical[1]
#           }

# indices = [6, 12, 18, 24]
# subplots_1d(x_values, values, indices, "CP within the Physical space - hopefully.")

# # %%

# %%
