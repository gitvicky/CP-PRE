#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc. 
Ideal MHD - Finite Volume Solver.

Parameterisation of calibration and prediction data: 
        lb = np.asarray([0.1, 0.1]) #a, b
        ub = np.asarray([0.5, 0.5])
"""

# %%
configuration = {"Case": 'MHD',
                 "Field": 'rho, u, v, p, Bx, By',
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
                 "T_in": 1,    
                 "T_out": 20,
                 "Step": 1,
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
import sys
sys.path.append("..")
from Neural_PDE.Models.FNO import *
from Neural_PDE.Utils.processing_utils import * 
from Neural_PDE.Utils.training_utils import * 
from Neural_PDE.UQ.inductive_cp import * 
from Utils.plot_tools import subplots_2d

#Setting up locations. 
file_loc = os.getcwd()
data_loc = os.path.dirname(os.getcwd()) + '/Neural_PDE/Data'
model_loc = file_loc + '/Weights'
plot_loc = file_loc + '/Plots'
#Setting up the seeds and devices
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
# %% 
#Setting up the simulations 
from pyDOE import lhs
from Neural_PDE.Numerical_Solvers.MHD.ConstrainedMHD_2D import * 

N = 128 #number of grid points
boxsize = 1.0 #domain size
tEnd = 0.5 #simulation timescale
dx = boxsize/N
dt = 1e-4

# rho, u, v, p, bx, by, dt, x, err = solve(N, boxsize, dt, a, b, c)


# %% 
# Utility Functions 
#Stacking the various fields for FNO. 
def stacked_fields(variables):
    stack = []
    for var in variables:
        var = torch.tensor(var, dtype=torch.float32) #Converting to Torch
        var = var.permute(0, 2, 3, 1) #Permuting to be BS, Nx, Ny, Nt
        stack.append(var)
    stack = torch.stack(stack, dim=1)
    return stack


def unstack_fields(field, axis, variable_names):
    fields = torch.split(field, 1, dim=axis)
    fields = [t.squeeze(axis) for t in fields]
    
    if len(fields) != len(variable_names):
        raise ValueError("Number of tensors and variable names should match.")
    
    variables = []
    for field in fields:
        variables.append(field.permute(0, 3, 1, 2))
    
    return variables

#Running the simulations. 
def gen_data(params):
    #Generating Data 
    Rho, uu, vv, pp, Bx, By = [], [], [], [], [], []

    for ii in tqdm(range(len(params))):
        rho, u, v, p, bx, by, dt, x, err = solve(N, boxsize, tEnd,  params[ii,0], params[ii,1], params[ii,2])
        
        Rho.append(rho)
        uu.append(u)
        vv.append(v)
        pp.append(p)
        Bx.append(bx)
        By.append(by)

    #Extraction
    t_slice = 25
    x_slice = 1

    rho = np.asarray(Rho)[:, ::t_slice, ::x_slice, ::x_slice]
    uu = np.asarray(uu)[:, ::t_slice, ::x_slice, ::x_slice]
    vv = np.asarray(vv)[:, ::t_slice, ::x_slice, ::x_slice]
    pp = np.asarray(pp)[:, ::t_slice, ::x_slice, ::x_slice]
    Bx = np.asarray(Bx)[:, ::t_slice, ::x_slice, ::x_slice]
    By = np.asarray(By)[:, ::t_slice, ::x_slice, ::x_slice]

    variables = [rho, uu, vv, pp, Bx, By]
    variables = stacked_fields(variables)

    return variables, x[::x_slice], dt*t_slice

#Generate Initial Conditions
def gen_ic(params):
    rho_ic, u_ic, v_ic, p_ic, bx_ic, by_ic = [], [], [], [], [], []

    for ii in tqdm(range(len(params))):
        rho, u, v, p, bx, by, dt, x, err = solve(N, boxsize, tEnd,  params[ii,0], params[ii,1], params[ii,2])

        rho_ic.append(rho[0])
        u_ic.append(u[0])
        v_ic.append(v[0])
        p_ic.append(p[0])
        bx_ic.append(bx[0])
        by_ic.append(by[0])

    rho_ic, u_ic, v_ic, p_ic, bx_ic, by_ic = np.asarray(rho_ic), np.asarray(u_ic), np.asarray(v_ic), np.asarray(p_ic), np.asarray(bx_ic), np.asarray(by_ic)

    variables_ic = [rho_ic, u_ic, v_ic, p_ic, bx_ic, by_ic]
    return stacked_fields(variables_ic)

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

    a = uu[..., :T_in]
    u = uu[..., T_in:T_out+T_in]

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
#Define Bounds
lb = np.asarray([0.5, 0.5, 0.5]) #a, b, c
ub = np.asarray([1.0, 1.0, 1.0])

dx = dx
dy = dx
dt = dt

gamma = 5/3 #Ideal Gas


from Utils.ConvOps_2d import ConvOperator
#Defining the required Convolutional Operations. 
D_t = ConvOperator(domain='t', order=1)#, scale=alpha)
D_x = ConvOperator(domain='x', order=1)#, scale=beta) 
D_y = ConvOperator(domain='y', order=1)#, scale=beta)
D_x_y = ConvOperator(domain=('x', 'y'), order=1)#, scale=beta)
D_xx_yy = ConvOperator(domain=('x','y'), order=2)#, scale=gamma)

#Continuity
def residual_continuity(vars, boundary=False):
    rho, u, v = vars[:,0], vars[:, 1], vars[:, 2]
    res = D_t(rho) + u*D_x(rho) + rho*D_x(u) + v*D_y(rho) + rho*D_y(v) 
    if boundary:
        return res
    else: 
        return res[...,1:-1,1:-1,1:-1]
    
#Momentum 
def residual_momentum(vars, boundary=False):
    rho, u, v, p, Bx, By = vars[:,0], vars[:, 1], vars[:, 2], vars[:, 3], vars[:, 4], vars[:, 5]

    res_x = D_t(u) + u*D_x(u) + (1/rho)*D_x(p) - 2*(Bx/rho)*D_x(Bx) + v*D_y(u) - (By/rho)*D_y(Bx) - (Bx/rho)*D_y(By)
    res_y = D_t(v) + u*D_x(v) + (1/rho)*D_y(p) - 2*(By/rho)*D_y(By) + v*D_y(v) - (By/rho)*D_x(Bx) - (Bx/rho)*D_x(By)

    if boundary:
        return res_x + res_y
    else: 
        return res_x[...,1:-1,1:-1,1:-1] + res_y[...,1:-1,1:-1,1:-1]


#Energy
def residual_energy(vars, boundary=False):
    rho, u, v, p, Bx, By = vars[:, 0], vars[:, 1], vars[:, 2], vars[:, 3], vars[:, 4], vars[:, 5]
    p_gas = p - 0.5*(Bx**2 + By**2)

    res = D_t(rho) + u*D_x(p) + v*D_y(p) + (gamma-2)*(u*Bx+v*By)*(D_x(Bx) + D_y(By)) + (gamma*p_gas+By**2)*D_x(u) + (gamma*p_gas+Bx**2)*D_y(v)- Bx*By*(D_y(u) + D_x(v))
       
    if boundary:
        return res
    else: 
        return res[...,1:-1,1:-1,1:-1]

#Induction
def residual_induction(vars, boundary=False):
    u, v, p, Bx, By = vars[:, 1], vars[:, 2], vars[:, 3], vars[:, 4], vars[:, 5]
    res_x = D_t(Bx) - By*D_y(u) + Bx*D_y(v) - v*D_y(Bx) + u*D_y(By) 
    res_y = D_t(By) + By*D_x(u) - Bx*D_x(v) - v*D_x(Bx) + u*D_x(By)

    if boundary:
        return res_x + res_y
    else: 
        return res_x[...,1:-1,1:-1,1:-1] + res_y[...,1:-1,1:-1,1:-1]


#Gauss Law
def residual_gauss(vars, boundary=False):
    Bx, By = vars[:, 4], vars[:, 5]
    res = D_x(Bx) + D_y(By)
    
    if boundary:
        return res
    else: 
        return res[...,1:-1,1:-1,1:-1]

# %% 
#Load the trained Model
model = FNO_multi2d(configuration['T_in'], configuration['Step'], configuration['Modes'], configuration['Modes'], configuration['Variables'], configuration['Width_time'])
model.load_state_dict(torch.load(model_loc + '/FNO_MHD_swift-method.pth', map_location='cpu'))
model.to(device)
print("Number of model params : " + str(model.count_params()))

#Loading normalisations 
norms = np.load(model_loc + '/FNO_MHD_swift-method_norms.npz')
in_normalizer, out_normalizer = normalisation(configuration['Normalisation Strategy'], norms)

# %% 
# Loading the Calibration Data
t1 = default_timer()
data =  np.load(data_loc + '/Constrained_MHD_combined.npz')

rho = data['rho'].astype(np.float32)
u = data['u'].astype(np.float32)
v = data['v'].astype(np.float32)
p = data['p'].astype(np.float32)
Bx = data['Bx'].astype(np.float32)
By  = data['By'].astype(np.float32)

x = data['x'].astype(np.float32)
y = data['x'].astype(np.float32)
t = data['t'].astype(np.float32)

vars = stacked_fields([rho, u, v, p, Bx, By])
field = ['rho', 'u', 'v', 'p', 'Bx', 'By']

# %%
#Calibration
cal_in, cal_out = data_loader(vars[:configuration['n_cal']], configuration['T_in'], configuration['T_out'], in_normalizer, out_normalizer, dataloader=False)
cal_pred, mse, mae = validation_AR(model, cal_in, cal_out, configuration['Step'], configuration['T_out'])
cal_out = out_normalizer.decode(cal_out)
cal_pred = out_normalizer.decode(cal_pred)

print('Calibration Error (MSE) : %.3e' % (mse))
print('Calibration Error (MAE) : %.3e' % (mae))

# %% 
#Calibration based on PRE: 

equation = 'continuity' #continuity, momentum, energy, induction, divergence

if equation == 'continuity':
# #Using the Continuity Equation. 
    cal_pred_residual = residual_continuity(cal_pred.permute(0,1,4,2,3)) 
    cal_out_residual = residual_continuity(cal_out.permute(0,1,4,2,3)) #Data-Driven

if equation == 'momentum':
# #Using the Momentum Equation. 
    cal_pred_residual = residual_momentum(cal_pred.permute(0,1,4,2,3)) 
    cal_out_residual = residual_momentum(cal_out.permute(0,1,4,2,3)) #Data-Driven

if equation == 'energy':
#Using Energy Equation
    cal_pred_residual = residual_energy(cal_pred.permute(0,1,4,2,3)) 
    cal_out_residual = residual_energy(cal_out.permute(0,1,4,2,3)) #Data-Driven

if equation == 'induction':
# Using Induction
    cal_pred_residual = residual_induction(cal_pred.permute(0,1,4,2,3)) 
    cal_out_residual = residual_induction(cal_out.permute(0,1,4,2,3)) #Data-Driven

if equation == 'divergence':
# #Using Gauss Law
    cal_pred_residual = residual_gauss(cal_pred.permute(0,1,4,2,3)) 
    cal_out_residual = residual_gauss(cal_out.permute(0,1,4,2,3)) #Data-Driven

modulation = modulation_func(cal_out_residual.numpy(), cal_pred_residual.numpy()) + 1e-6
ncf_scores = ncf_metric_joint(cal_out_residual.numpy(), cal_pred_residual.numpy(), modulation)

# %%
#Checking for coverage from a portion of the available data
pred_in, pred_out = data_loader(vars[-configuration['n_pred']:], configuration['T_in'], configuration['T_out'], in_normalizer, out_normalizer, dataloader=False)
pred_pred, mse, mae = validation_AR(model, pred_in, pred_out, configuration['Step'], configuration['T_out'])
pred_out = out_normalizer.decode(pred_out)
pred_pred = out_normalizer.decode(pred_pred)

# %%

if equation == 'continuity':
# #Using the Continuity Equation. 
    pred_residual = residual_continuity(pred_pred.permute(0,1,4,2,3)) #Prediction
    val_residual = residual_continuity(pred_out.permute(0,1,4,2,3)) #Data

if equation == 'momentum':
# #Using the Momentum Equation. 
    pred_residual = residual_momentum(pred_pred.permute(0,1,4,2,3)) #Prediction
    val_residual = residual_momentum(pred_out.permute(0,1,4,2,3)) #Data

if equation == 'energy':
#Using Energy Equation
    pred_residual = residual_energy(pred_pred.permute(0,1,4,2,3)) #Prediction
    val_residual = residual_energy(pred_out.permute(0,1,4,2,3)) #Data

if equation == 'induction':
# Using Induction
    pred_residual = residual_induction(pred_pred.permute(0,1,4,2,3)) #Prediction
    val_residual = residual_induction(pred_out.permute(0,1,4,2,3)) #Data

if equation == 'divergence':
# #Using Gauss Law
    pred_residual = residual_gauss(pred_pred.permute(0,1,4,2,3)) #Prediction
    val_residual = residual_gauss(pred_out.permute(0,1,4,2,3)) #Data


#Emprical Coverage for all values of alpha 
alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov_res = []
for alpha in tqdm(alpha_levels):
    qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
    prediction_sets = [pred_residual.numpy() - qhat*modulation, pred_residual.numpy() + qhat*modulation]
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
# res = cal_out_residual #Data-Driven
res = cal_pred_residual #Physics-Driven

modulation = modulation_func(res.numpy(), np.zeros(res.shape))
ncf_scores = ncf_metric_joint(res.numpy(), np.zeros(res.shape), modulation)

#Emprical Coverage for all values of alpha to see if pred_residual lies between +- qhat. 
# alpha_levels = np.arange(0.05, 0.95, 0.1)
alpha_levels = np.arange(0.05, 0.95+0.1, 0.1)

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

#Plotting the error bars. 
idx = 5
t_idx = 10
values = [
          prediction_sets[0][t_idx],
          prediction_sets[1][t_idx]
          ]

titles = [
          r'$- \hat q $',
          r'$+ \hat q $'
          ]

subplots_2d(values, titles)
# %%
#Paper Plots 

import matplotlib as mpl 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import matplotlib.ticker as ticker

alpha = 0.1
qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
prediction_sets = [-qhat*modulation, + qhat*modulation]


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

idx = 20
t_idx= 15


# Create figure and axis
fig, ax = plt.subplots()

# Plot the image
im = ax.imshow(pred_residual[idx, t_idx], cmap='magma')

# Create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.15)

# Create colorbar in the appended axes
cbar = plt.colorbar(im, cax=cax)
# Set colorbar ticks to use scientific notation
cbar.formatter = ticker.ScalarFormatter(useMathText=True)
cbar.formatter.set_scientific(True)
cbar.formatter.set_powerlimits((0, 0))
cbar.update_ticks()
cbar.ax.tick_params(labelsize=36)

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])

# Set labels and title
ax.set_xlabel(r'$x$', fontsize=36)
ax.set_ylabel(r'$y$', fontsize=36)
# ax.set_title(r'PRE: $D_{induction}(\vec{v},\vec{B})$', fontsize=36)

# plt.savefig(os.path.dirname(os.getcwd()) + "/Plots/mhd_residual_" + equation + "_.svg", format="svg",transparent=True, bbox_inches='tight')
# plt.savefig(os.path.dirname(os.getcwd()) + "/Plots/mhd_residual_" + equation + "_.pdf", format="pdf",transparent=True, bbox_inches='tight')
plt.show()


# Create figure and axis
fig, ax = plt.subplots()

# Plot the image
im = ax.imshow(prediction_sets[1][t_idx], cmap='magma')

# Create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.15)

# Create colorbar in the appended axes
cbar = plt.colorbar(im, cax=cax)
# Set colorbar ticks to use scientific notation
cbar.formatter = ticker.ScalarFormatter(useMathText=True)
cbar.formatter.set_scientific(True)
cbar.formatter.set_powerlimits((0, 0))
cbar.update_ticks()
cbar.ax.tick_params(labelsize=36)

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])

# Set labels and title
ax.set_xlabel(r'$x$', fontsize=36)
ax.set_ylabel(r'$y$', fontsize=36)
ax.set_title(r'Joint CP ($+\hat q \times mod)$', fontsize=36)

plt.savefig(os.path.dirname(os.getcwd()) + "/Plots/joint_mhd_" + equation + "_qhat.svg", format="svg", transparent=True, bbox_inches='tight')
plt.savefig(os.path.dirname(os.getcwd()) + "/Plots/joint_mhd_" + equation + "_qhat.pdf", format="pdf", transparent=True, bbox_inches='tight')

plt.show()

# %%
