#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uncertainty quantification of a Neural PDE Surrogate solving the Advection Equation using Physics Residuals and guaranteed using Conformal Prediction
Prediction Selection/Rejection based on CP bounds. 

Equation :  U_t + v U_x = 0
Surrogate Model : FNO
Numerical Solver : https://github.com/gitvicky/Neural_PDE/blob/main/Numerical_Solvers/Advection/Advection_1D.py

Two attempts are explored: Using the AER over the PRE and then PRE alone.
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
                 "T_out": 20,
                 "Step": 1,
                 "Width": 16, 
                 "Modes": 8,
                 "Variables":1, 
                 "Noise":0.0, 
                 "Loss Function": 'MSE',
                 "n_train": 100,
                 "n_test": 1000,
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
from Neural_PDE.UQ.inductive_cp import * 

#Setting up locations. 
file_loc = os.getcwd()

#Setting up the seeds and devices
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

# %% 
#Extracting configuration files
T_in = configuration['T_in']
T_out = configuration['T_out']
step = configuration['Step']
width = configuration['Width']
modes = configuration['Modes']
output_size = configuration['Step']
num_vars = configuration['Variables']
batch_size = configuration['Batch Size']

# %% 
#Simulation Setup
from Neural_PDE.Numerical_Solvers.Advection.Advection_1D import *
from pyDOE import lhs

#Obtaining the exact and FD solution of the 1D Advection Equation. 
Nx = 200 #Number of x-points
Nt = 100 #Number of time instances 
x_min, x_max = 0.0, 2.0 #X min and max
t_end = 0.5 #time length
v = 1.0
sim = Advection_1d(Nx, Nt, x_min, x_max, t_end) 
dt, dx = sim.dt, sim.dx
# %% 
#Utility Functions

def gen_data(params):
    #Generating Data 
    u_sol = []
    for ii in tqdm(range(len(params))):
        xc = params[ii, 0]
        amp = params[ii, 1]
        x, t, u_soln, u_exact = sim.solve(xc, amp, v)
        u_sol.append(u_soln)

    #Extraction
    u_sol = np.asarray(u_sol)
    u_sol = u_sol[:, :, 1:-2]
    x = x[1:-2]

    #Tensorize
    u = torch.tensor(u_sol, dtype=torch.float32)
    u = u.permute(0, 2, 1) #only for FNO
    u = u.unsqueeze(1)

    return x, t, u

#Generate Initial Conditions
def gen_ic(params):
    u_ic = []
    for ii in tqdm(range(len(params))):
        xc = params[ii, 0]
        amp = params[ii, 1]
        sim.initializeU(xc, amp)
        u_ic.append(sim.u)

    u_ic = np.asarray(u_ic)[:, 1:-2]

    u_ic = torch.tensor(u_ic, dtype=torch.float32).unsqueeze(1).unsqueeze(-1)
    return u_ic

#Load Simulation data into Dataloader
def data_loader(uu, dataloader=True, shuffle=True):

    a = uu[:, :, :, :T_in]
    u = uu[:, :, :, T_in:T_out+T_in]

    # print("Input: " + str(a.shape))
    # print("Output: " + str(u.shape))

    #No Normalisation -- Normalisation = Identity 

    if dataloader:
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(a, u), batch_size=batch_size, shuffle=shuffle)
    else:
        loader = [a,u]

    return loader

# %% 
#Define Bounds
lb = np.asarray([0.5, 50]) #pos, amplitude
ub = np.asarray([1.0, 200])

#Conv kernels --> Stencils 
from Utils.ConvOps_1d import ConvOperator
#Defining the required Convolutional Operations. 
D_t = ConvOperator(domain='t', order=1)#, scale=alpha)
D_x = ConvOperator(domain='x', order=1)#, scale=beta) 

#Residual - Additive Kernels
disc = 2
D = ConvOperator()
D.kernel = D_t.kernel + (v*disc*dt/dx) * D_x.kernel

################################################################
#Train Data
params = lb + (ub - lb) * lhs(2, configuration['n_train'])
x, t, u_sol = gen_data(params)
train_loader = data_loader(u_sol)
test_loader = data_loader(u_sol[-10:]) #Just kept to hefty evaluations each epoch. 

#Test Data
params = lb + (ub - lb) * lhs(2, configuration['n_test'])
x, t, u_sol = gen_data(params)
test_a, test_u = data_loader(u_sol, dataloader=False, shuffle=False)
test_mse = []

#Initialising the model
model = FNO_multi1d(T_in, step, modes, num_vars, width, width_vars=0)
model.to(device)
print("Number of model params : " + str(model.count_params()))

#Setting up the optimizer and scheduler, loss and epochs 
optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
loss_func = torch.nn.MSELoss()
epochs = configuration['Epochs']

start_time = default_timer()
####################################
#Training Loop 
####################################
for ep in range(epochs): #Training Loop - Epochwise

    model.train()
    t1 = default_timer()
    train_loss, test_loss = train_one_epoch_AR(model, train_loader, test_loader, loss_func, optimizer, step, T_out)
    t2 = default_timer()

    # train_loss = train_loss / ntrain / num_vars
    # test_loss = test_loss / ntest / num_vars

    print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 3)}, Test Loss: {round(test_loss,3)}")
    
    scheduler.step()
    train_time = default_timer() - start_time

#Evaluation
pred_test, mse, mae = validation_AR(model, test_a, test_u, step, T_out)
test_mse.append(mse)
print('Testing Error (MSE) : %.3e' % (mse))
print('Testing Error (MAE) : %.3e' % (mae))
print()

# %% 
# #Calibration Residuals with just prediction data
# params = lb + (ub - lb) * lhs(2, configuration['n_cal'])
# u_in_cal = gen_ic(params)
# pred_cal, mse, mae = validation_AR(model, u_in_cal, torch.zeros((u_in_cal.shape[0], u_in_cal.shape[1], u_in_cal.shape[2], T_out)), configuration['Step'], configuration['T_out'])
# uu_cal = pred_cal.permute(0,1,3,2)[:,0]
# cal_residual = D(uu_cal)
# ncf_scores = np.abs(cal_residual.numpy()) 

## Using Calibration Data from smaller sample of simulations
params = lb + (ub - lb) * lhs(2, configuration['n_cal'])
x, t, u_sol = gen_data(params)
u_in_cal, u_out_cal = data_loader(u_sol, dataloader=False, shuffle=False)
u_pred_cal, mse, mae = validation_AR(model, u_in_cal, u_out_cal, step, T_out)

u_out_cal = u_out_cal[...,::disc]
u_pred_cal = u_pred_cal[...,::disc]

residual_out_cal = D(u_out_cal.permute(0,1,3,2)[:,0])[...,1:-1, 1:-1]
residual_pred_cal = D(u_pred_cal.permute(0,1,3,2)[:,0])[...,1:-1, 1:-1]

ncf_scores = np.abs(residual_out_cal - residual_pred_cal) #AER + PRE
# %%
#Prediction Residuals 
params = lb + (ub - lb) * lhs(2, configuration['n_pred'])
u_in_pred = gen_ic(params)
pred_pred, mse, mae = validation_AR(model, u_in_pred, torch.zeros((u_in_pred.shape[0], u_in_pred.shape[1], u_in_pred.shape[2], T_out)), configuration['Step'], configuration['T_out'])
pred_pred = pred_pred.permute(0,1,3,2)[:,0]
uu_pred = pred_pred
uu_pred = uu_pred[:,::disc]
pred_residual = D(uu_pred)[...,1:-1, 1:-1]

# %% 
#Plotting Coverage
alpha = 0.1
qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
prediction_sets =  [pred_residual.numpy() - qhat, pred_residual.numpy() + qhat]

from Utils.plot_tools import subplots_1d
x_values = x[1:-1]
idx = 5
values = {"Residual": pred_residual[idx], 
          "Lower": prediction_sets[0][idx],
          "Upper": prediction_sets[1][idx]
          }

indices = [2, 3, 4, 5]
subplots_1d(x_values, values, indices, "CP within the residual space.")

# %% 
#Checking for Coverage
pred_test, mse, mae = validation_AR(model, test_a, test_u, step, T_out)

test_val_residual = D(test_u.permute(0,1,3,2)[:,0])[...,1:-1, 1:-1]
test_pred_residual = D(pred_test.permute(0,1,3,2)[:,0])[...,1:-1, 1:-1]

#Emprical Coverage for all values of alpha 
alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov_res = []
for alpha in tqdm(alpha_levels):
    qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
    prediction_sets = [test_pred_residual.numpy() - qhat, test_pred_residual.numpy() + qhat]
    emp_cov_res.append(emp_cov(prediction_sets, test_val_residual.numpy()))

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()

# %% 
#Plotting Coverage
alpha = 0.5
qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
prediction_sets =  [- qhat, + qhat]

from Utils.plot_tools import subplots_1d
x_values = x[1:-1]
idx = 5
values = {"Residual": pred_residual[idx],
          "Lower": prediction_sets[0],
          "Upper": prediction_sets[1]
          }

indices = [1, 3, 5, 7]
subplots_1d(x_values, values, indices, "Marginal CP (50% coverage)")
# plt.savefig(os.path.dirname(os.getcwd()) + "/Plots/marginal_advection_" + str(alpha) + "_.pdf", format="pdf", bbox_inches='tight')


#%%
###################################################################
# Data Free -- using PRE only 
# ncf_scores = np.abs(residual_out_cal) #Data-Driven
ncf_scores = np.abs(residual_pred_cal) #Physics-Driven
###################################################################

#Emprical Coverage for all values of alpha to see if pred_residual lies between +- qhat. 
alpha_levels = np.arange(0.05, 0.95+0.1, 0.1)
emp_cov_res = []
for alpha in tqdm(alpha_levels):
    qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
    prediction_sets = [- qhat, + qhat]
    emp_cov_res.append(emp_cov(prediction_sets, pred_residual.numpy()))

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()

# %%
#Selection/Rejection
alpha = 0.1
threshold = 0.5
qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
prediction_sets = [- qhat,  + qhat]

from Utils.plot_tools import subplots_1d
x_values = x[1:-1]
idx = 40
values = {"Residual": pred_residual[idx], 
          "Lower": prediction_sets[0],
          "Upper": prediction_sets[1]
          }

indices = [1, 3, 5, 7]
subplots_1d(x_values, values, indices, "CP within the residual space.")

filtered_sims = filter_sims_within_bounds(prediction_sets[0], prediction_sets[1], pred_residual.numpy(), threshold=threshold)
params_filtered = params[filtered_sims]
print(f'{len(params_filtered)} simulations rejected')

# %%
#Paper Plots 
alpha = 0.1
qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
prediction_sets = [- qhat,  + qhat]

import matplotlib as mpl 
# Set matplotlib parameters
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['font.size'] = 24
mpl.rcParams['figure.figsize'] = (9,9)
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['axes.titlepad'] = 20
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['xtick.minor.size'] = 5.0
plt.rcParams['ytick.minor.size'] = 5.0
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.minor.width'] = 0.6
plt.rcParams['ytick.minor.width'] = 0.6
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '-'

idx = 5
t_idx = 3

plt.plot(x_values, pred_residual[idx, t_idx], label='PRE', color='black',lw=4, ls='--', alpha=0.75)
plt.plot(x_values, prediction_sets[0][t_idx], label='Lower Bound', color='maroon',lw=4, ls='--',  alpha=0.75)
plt.plot(x_values, prediction_sets[1][t_idx], label='Upper Bound', color='red',lw=4, ls='--',  alpha=0.75)

plt.xlabel(r'$x$', fontsize=36)
plt.ylabel(r'$D(u)$', fontsize=36)

# Customize x-axis ticks
plt.xticks( # 5 ticks from min to max
    fontsize=36  # Increase font size
)
plt.yticks( # 5 ticks from min to max
        # np.linspace(-0.015, 0.015, 5),
    fontsize=36  # Increase font size
)

plt.title(r"Coarse ($dt$=0.01)", fontsize=36)
plt.legend(fontsize=24)
# plt.savefig(os.path.dirname(os.getcwd()) + "/Plots/marginal_advection.svg", format="svg", bbox_inches='tight')
plt.savefig(os.path.dirname(os.getcwd()) + "/Plots/marginal_advection_coarse.pdf", format="pdf", bbox_inches='tight')
plt.show()
# %%

#Paper Plots - Rebuttal 
alpha = 0.1
qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
prediction_sets = [- qhat,  + qhat]

import matplotlib as mpl 
# Set matplotlib parameters
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['font.size'] = 24
mpl.rcParams['figure.figsize'] = (9,9)
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['axes.titlepad'] = 20
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['xtick.minor.size'] = 5.0
plt.rcParams['ytick.minor.size'] = 5.0
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.minor.width'] = 0.6
plt.rcParams['ytick.minor.width'] = 0.6
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '-'

idx = 5
t_idx = 5

plt.plot(x, test_u[idx, 0, :, t_idx-1], label='Ground Truth', color='black',lw=4, ls='--', alpha=0.75)
plt.plot(x, pred_test[idx, 0, :, t_idx-1], label='PRE-CP', color='blue',lw=4, ls='--', alpha=0.75)
# plt.plot(x, pred_test_bad[idx, 0, :, t_idx-1], label='Bad Model', color='red',lw=4, ls='--', alpha=0.75)

plt.xlabel(r'$x$', fontsize=36)
plt.ylabel(r'$D(u)$', fontsize=36)

# Customize x-axis ticks
plt.xticks( # 5 ticks from min to max
    fontsize=36  # Increase font size
)
plt.yticks( # 5 ticks from min to max
        # np.linspace(-0.002, 0.002, 5),
    fontsize=36  # Increase font size
)

plt.title("Model Performance", fontsize=36)
plt.legend(fontsize=24)
# plt.savefig(os.path.dirname(os.getcwd()) + "/Plots/marginal_advection_goodbad.svg", format="svg", bbox_inches='tight')
# plt.savefig(os.path.dirname(os.getcwd()) + "/Plots/marginal_advection_goodbad.pdf", format="pdf", bbox_inches='tight')
plt.show()

# %%

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, color='black', alpha=0.6, lw=5.0, label='Ideal')
plt.plot(1-alpha_levels, emp_cov_coarse, color='maroon', ls='--', alpha=0.6, lw=4.0, label='Coarse')
plt.plot(1-alpha_levels, emp_cov_res, ls='-.', color='navy', alpha=0.8, lw=3.0, label='Fine')
plt.xlabel(r'1-$\alpha$', fontsize=36)
plt.ylabel(r'Empirical Coverage', fontsize=36)
plt.title('Coverage', fontsize=36)
plt.legend(fontsize=24)

# plt.savefig(os.path.dirname(os.getcwd()) + "/Plots/marginal_advection_coverage.svg", format="svg", bbox_inches='tight')
plt.savefig(os.path.dirname(os.getcwd()) + "/Plots/marginal_advection_disc_coverage.pdf", format="pdf", bbox_inches='tight')
plt.show()
# %%
