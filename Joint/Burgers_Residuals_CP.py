#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uncertainty quantification of a Neural PDE Surrogate solving the Burgers Equation using Physics Residuals 
and guaranteed using Conformal Prediction. Prediction Selection/Rejection based on CP bounds. 

Eqn:   u_t + u*u_x =  nu*u_xx on [0,2]

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
                 "Normalisation Strategy": 'Min-Max',
                 "T_in": 1,    
                 "T_out": 30,
                 "Step": 1,
                 "Width": 32, 
                 "Modes": 8,
                 "Variables":1, 
                 "Noise":0.0, 
                 "Loss Function": 'MSE',
                 "n_train": 100,
                 "n_test": 1000,
                 "n_cal": 1000,
                 "n_pred": 100
                 }
# %% 
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
from Utils.plot_tools import subplots_2d

#Setting up locations. 
file_loc = os.getcwd()
# data_loc = file_loc + '/Data'
model_loc = file_loc + '/Weights'
plot_loc = file_loc + '/Plots'
#Setting up the seeds and devices
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

# %% 
# Generating calibration data
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
sim = Burgers_1D(Nx, Nt, x_min, x_max, t_end, nu) 
dt, dx = sim.dt, sim.dx
dx, dt = dx*x_slice, dt*t_slice
# %% 
# Utility Functions 

def gen_data(params):
    #Generating Data 
    u_sol = []
    for ii in tqdm(range(len(params))):
        sim.InitializeU(params[ii,0], params[ii,1], params[ii,2])
        u_soln, x, dt = sim.solve()
        u_sol.append(u_soln)

    #Extraction
    u_sol = np.asarray(u_sol)[:, ::t_slice, ::x_slice]
    x = x[::x_slice]
    dt = dt*t_slice

    #Tensorize
    u = torch.tensor(u_sol, dtype=torch.float32)
    u = u.permute(0, 2, 1) #Adding BS and Permuting for FNO
    u = u.unsqueeze(1) #Adding the variable channel

    return x, dt, u


#Generate Initial Conditions
def gen_ic(params):
    u_ic = []
    for ii in tqdm(range(len(params))):
        sim.InitializeU(params[ii,0], params[ii,1], params[ii,2])
        u_ic.append(sim.u0)

    u_ic = np.asarray(u_ic)[:, ::x_slice]
    u_ic = torch.tensor(u_ic, dtype=torch.float32).unsqueeze(1).unsqueeze(-1)
    return u_ic

def normalisation(norm_strategy, norms):
    if norm_strategy == 'Min-Max':
        normalizer = MinMax_Normalizer
    elif norm_strategy == 'Range':
        normalizer = RangeNormalizer
    elif norm_strategy == 'Gaussian':
        normalizer = GaussianNormalizer
    elif norm_strategy == 'Identity':
        normalizer = Identity

    #Loading the Normaliation values
    in_normalizer = MinMax_Normalizer(torch.tensor(0))
    in_normalizer.a = torch.tensor(norms['in_a'])
    in_normalizer.b = torch.tensor(norms['in_b'])

    out_normalizer = MinMax_Normalizer(torch.tensor(0))
    out_normalizer.a = torch.tensor(norms['in_a'])
    out_normalizer.b = torch.tensor(norms['in_b'])
    
    return in_normalizer, out_normalizer


#Load Simulation data into Dataloader
def data_loader(uu, T_in, T_out, in_normalizer, out_normalizer, dataloader=True, shuffle=True):

    a = uu[:, :, :, :T_in]
    u = uu[:, :, :, T_in:T_out+T_in]

    # print("Input: " + str(a.shape))
    # print("Output: " + str(u.shape))

    #Performing the Normalisation and Setting up the DataLoaders
    a = in_normalizer.encode(a)
    u  = out_normalizer.encode(u)

    if dataloader:
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(a, u), batch_size=configuration['Batch Size'], shuffle=shuffle)
    #Performing the normalisation on the input alone. 
    else:
        loader = [a,u]
    return loader

# %% 
# Define Bounds
lb = np.asarray([-3, -3, -3]) # Lower Bound of the parameter domain
ub = np.asarray([3, 3, 3]) # Upper bound of the parameter domain

#Conv kernels --> Stencils 
from Utils.ConvOps_1d import ConvOperator
#Defining the required Convolutional Operations. 
D_t = ConvOperator(domain='t', order=1)#, scale=alpha)
D_x = ConvOperator(domain='x', order=1)#, scale=beta) 
D_xx = ConvOperator(domain='x', order=2)#, scale=gamma)

dx = torch.tensor(dx, dtype=torch.float32)
dt = torch.tensor(dt, dtype=torch.float32)
nu = torch.tensor(nu, dtype=torch.float32)

# Residual
def residual(uu, boundary=False):
    res = dx*D_t(uu) + dt * uu * D_x(uu) - nu * D_xx(uu) * (2*dt/dx)
    if boundary:
        return res
    else: 
        return res[...,1:-1,1:-1]

#Loading the Model  
model = FNO_multi1d(configuration['T_in'], configuration['Step'], configuration['Modes'], configuration['Variables'], configuration['Width'], width_vars=0)
model.to(device)
print("Number of model params : " + str(model.count_params()))
model.load_state_dict(torch.load(model_loc + '/FNO_Burgers_worn-insulation.pth', map_location='cpu'))

# %% 
################################################################
#Loading Calibration Data
data_loc = os.path.dirname(os.getcwd()) + '/Neural_PDE/Data'
data =  np.load(data_loc + '/Burgers_1d.npz')

u_sol  = data['u']
x = data['x']
# dt = data['dt']
u = torch.tensor(u_sol, dtype=torch.float32)
u = u.permute(0, 2, 1) #Adding BS and Permuting for FNO
u = u.unsqueeze(1) #Adding the variable channel

#Setting up the normalisation. 
norms = np.load(model_loc + '/FNO_Burgers_worn-insulation_norms.npz')
in_normalizer, out_normalizer = normalisation(configuration['Normalisation Strategy'], norms)

cal_in, cal_out = data_loader(u[:500], configuration['T_in'], configuration['T_out'], in_normalizer, out_normalizer, dataloader=False)
cal_pred, mse, mae = validation_AR(model, cal_in, cal_out, configuration['Step'], configuration['T_out'])
cal_out = out_normalizer.decode(cal_out)
cal_pred = out_normalizer.decode(cal_pred)

cal_pred_residual = residual(cal_pred.permute(0,1,3,2)[:,0]) #Physics-Driven
cal_out_residual = residual(cal_out.permute(0,1,3,2)[:,0])#Data-Driven
modulation = modulation_func(cal_out_residual.numpy(), cal_pred_residual.numpy())
ncf_scores = ncf_metric_joint(cal_pred_residual.numpy(), cal_out_residual.numpy(), modulation)

# %% 
# from Utils.plot_tools import subplots_1d
# x_values = x[1:-1]
# idx = 40
# values = {"Target": cal_out[idx][0].T, 
#           "Pred.": cal_pred[idx][0].T,
#           }

# indices = [5, 10, 15, 20]
# subplots_1d(x_values, values, indices, "Plotting NN performance across calibration space.")

# # %%
# x_values = x[1:-1]
# idx = 40
# values = {"Residual: Targ.": residual(cal_out.permute(0,1,3,2)[:,0])[idx], 
#           "Residual: Pred.": residual(cal_pred.permute(0,1,3,2)[:,0])[idx],
#           }

# indices = [5, 10, 15, 20]
# subplots_1d(x_values, values, indices, "Residual Comparison")

# %% 
#Checking for coverage from a portion of the available data
pred_in, pred_out = data_loader(u[500:], configuration['T_in'], configuration['T_out'], in_normalizer, out_normalizer, dataloader=False)
pred_pred, mse, mae = validation_AR(model, pred_in, pred_out, configuration['Step'], configuration['T_out'])
pred_out = out_normalizer.decode(pred_out)
pred_pred = out_normalizer.decode(pred_pred)

pred_residual = residual(pred_pred.permute(0,1,3,2)[:,0]) #Prediction
val_residual = residual(pred_out.permute(0,1,3,2)[:,0]) #Data

#Emprical Coverage for all values of alpha 
alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov_res = []
for alpha in tqdm(alpha_levels):
    qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
    prediction_sets = [pred_residual.numpy() - qhat*modulation,  pred_residual.numpy() +  qhat*modulation]
    emp_cov_res.append(emp_cov_joint(prediction_sets, val_residual.numpy()))

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()

# %% 
###################################################################
#Filtering Sims -- using PRE only 
res = cal_out_residual #Data-Driven
# res = cal_pred_residual #Physics-Driven

modulation = modulation_func(res.numpy(), np.zeros(res.shape))
ncf_scores = ncf_metric_joint(res.numpy(), np.zeros(res.shape), modulation)

#Emprical Coverage for all values of alpha to see if pred_residual lies between +- qhat. 
alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov_res = []
for alpha in tqdm(alpha_levels):
    qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
    prediction_sets = [- qhat*modulation, + qhat*modulation]
    emp_cov_res.append(emp_cov_joint(prediction_sets, pred_residual.numpy()))

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()

# %% 
###################################################################
#Filtering Sims

def filter_sims_joint(prediction_sets, y_response):
    axes = tuple(np.arange(1,len(y_response.shape)))
    return ((y_response >= prediction_sets[0]).all(axis = axes) & (y_response <= prediction_sets[1]).all(axis = axes))

alpha = 0.5
qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
prediction_sets =  [- qhat*modulation, + qhat*modulation]
filtered_sims = filter_sims_joint(prediction_sets, pred_residual.numpy())
print(filtered_sims)
print(f'{sum(filtered_sims)} simulations rejected')
