#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desc. 
Incompressible Navier-Stokes - Finite Volume Solver.

Parameterisation of calibration and prediction data: 
        lb = np.asarray([0.1, 0.1]) #a, b
        ub = np.asarray([0.5, 0.5])
"""

# %%
configuration = {"Case": 'Navier-Stokes',
                 "Field": 'u, v, p, w',
                 "Model": 'FNO',
                 "Epochs": 500,
                 "Batch Size": 5,
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
from Neural_PDE.Numerical_Solvers.Navier_Stokes.NS_2D_spectral import * 

N = 400 #Number of grid points
tStart = 0.0 #Starting time of the simulation
tEnd = 0.5 #Simulation ending time
dt = 0.001 #dt
nu = 0.001#kinematic viscosity
L = 1 #Domain Length
aa = 1.0#parametrisation of initial Vx 
bb = 1.0#parametrisation of initial Vx 
solver= Navier_Stokes_2d(N, tStart, dt, dt, nu, L, aa, bb)
u, v, p, w, x, t, err = solver.solve()

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
    uu, vv, pp = [], [], []

    for ii in tqdm(range(len(params))):
        sim = Navier_Stokes_2d(N, tStart, tEnd, dt, nu, L, params[ii,0], params[ii,1])
        u, v, p, w, x, t, err = sim.solve()
        uu.append(u)
        vv.append(v)
        pp.append(p)

    #Extraction
    t_slice = 10 
    x_slice = 4
    uu = np.asarray(uu)[:, ::t_slice, ::x_slice, ::x_slice]
    vv = np.asarray(vv)[:, ::t_slice, ::x_slice, ::x_slice]
    pp = np.asarray(pp)[:, ::t_slice, ::x_slice, ::x_slice]

    variables = [uu, vv, pp]
    variables = stacked_fields(variables)

    return variables, x[::x_slice], t[::t_slice]

#Generate Initial Conditions
def gen_ic(params):
    u_ic, v_ic, p_ic = [], [], []

    for ii in tqdm(range(len(params))):
        sim= Navier_Stokes_2d(N, tStart, tEnd, dt, nu, L, params[ii,0], params[ii,1])
        solver= Navier_Stokes_2d(N, tStart, dt, dt, nu, L, aa, bb)
        u, v, p, w, x, t, err = solver.solve()

        u_ic.append(u[0])
        v_ic.append(v[0])
        p_ic.append(p[0])

    u_ic, v_ic, p_ic = np.asarray(u_ic), np.asarray(v_ic), np.asarray(p_ic)

    variables_ic = [u_ic, v_ic, p_ic]
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
    out_normalizer.a = torch.tensor(norms['out_a'])
    out_normalizer.b = torch.tensor(norms['out_b'])
    
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
lb = np.asarray([0.5, 0.5]) #Vx - aa, Vy - bb
ub = np.asarray([1.0, 1.0])

dx = np.asarray(x[-1] - x[-2])
dy = dx
dt = dt

alpha = 1/dt*2
beta = 1/dx*2
gamma = 1/dx**2                 

alpha, beta, gamma = 1,1,1

from Utils.ConvOps_2d import ConvOperator
#Defining the required Convolutional Operations. 
D_t = ConvOperator(domain='t', order=1)#, scale=alpha)
D_x = ConvOperator(domain='x', order=1)#, scale=beta) 
D_y = ConvOperator(domain='y', order=1)#, scale=beta)
D_x_y = ConvOperator(domain=('x', 'y'), order=1)#, scale=beta)
D_xx_yy = ConvOperator(domain=('x','y'), order=2)#, scale=gamma)

#Continuity
def residual_continuity(vars, boundary=False):
    u, v = vars[:,0], vars[:, 1]
    res = D_x(u) + (dx/dy) * D_y(v)
    if boundary:
        return res
    else: 
        return res[...,1:-1,1:-1,1:-1]
    
#Momentum 
def residual_momentum(vars, boundary=False):
    u, v, p = vars[:,0], vars[:, 1], vars[:, 2]

    res_x = D_t(u)*dx*dy + u*D_x(u)*dt*dy + v*D_y(u)*dt*dx - nu*D_xx_yy(u)*dt + D_x(p)*dt*dy
    res_y = D_t(v)*dx*dy + u*D_x(v)*dt*dx + v*D_y(v)*dt*dy - nu*D_xx_yy(v)*dt + D_y(p)*dt*dx

    if boundary:
        return res_x + res_y
    else: 
        return res_x[...,1:-1,1:-1,1:-1] + res_y[...,1:-1,1:-1,1:-1]
    

# %% 
#Load the trained Model
model = FNO_multi2d(configuration['T_in'], configuration['Step'], configuration['Modes'], configuration['Modes'], configuration['Variables'], configuration['Width_time'])
model.load_state_dict(torch.load(model_loc + '/FNO_Navier-Stokes_violent-remote.pth', map_location='cpu'))
model.to(device)
print("Number of model params : " + str(model.count_params()))

#Loading normalisations 
norms = np.load(model_loc + '/FNO_Navier-Stokes_violent-remote_norms.npz')
in_normalizer, out_normalizer = normalisation(configuration['Normalisation Strategy'], norms)

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

vars = stacked_fields([u,v,p,w])

field = ['u', 'v', 'p', 'w']

cal_in, cal_out = data_loader(vars[:configuration['n_cal']], configuration['T_in'], configuration['T_out'], in_normalizer, out_normalizer, dataloader=False)
cal_pred, mse, mae = validation_AR(model, cal_in, cal_out, configuration['Step'], configuration['T_out'])
cal_out = out_normalizer.decode(cal_out)
cal_pred = out_normalizer.decode(cal_pred)

print('Calibration Error (MSE) : %.3e' % (mse))
print('Calibration Error (MAE) : %.3e' % (mae))

# %% 
#Using the Continuity Equation. 
cal_pred_residual = residual_continuity(cal_pred.permute(0,1,4,2,3)) 
cal_out_residual = residual_continuity(cal_out.permute(0,1,4,2,3)) #Data-Driven

#Using the Momentum Equation. 
cal_pred_residual = residual_momentum(cal_pred.permute(0,1,4,2,3)) 
cal_out_residual = residual_momentum(cal_out.permute(0,1,4,2,3)) #Data-Driven

modulation = modulation_func(cal_out_residual.numpy(), cal_pred_residual.numpy())
ncf_scores = ncf_metric_joint(cal_out_residual.numpy(), cal_pred_residual.numpy(), modulation)

# %%
#Checking for coverage from a portion of the available data
pred_in, pred_out = data_loader(vars[-configuration['n_pred']:], configuration['T_in'], configuration['T_out'], in_normalizer, out_normalizer, dataloader=False)
pred_pred, mse, mae = validation_AR(model, pred_in, pred_out, configuration['Step'], configuration['T_out'])
pred_out = out_normalizer.decode(pred_out)
pred_pred = out_normalizer.decode(pred_pred)

#Using the Continuity Equation. 
pred_residual = residual_continuity(pred_pred.permute(0,1,4,2,3)) #Prediction
val_residual = residual_continuity(pred_out.permute(0,1,4,2,3)) #Data

#Using the Momentum Equation. 
pred_residual = residual_momentum(pred_pred.permute(0,1,4,2,3)) #Prediction
val_residual = residual_momentum(pred_out.permute(0,1,4,2,3)) #Data

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

# %% 

#Plotting the error bars. 
idx = 5
t_idx = 10
values = [
          prediction_sets[0][t_idx],
          prediction_sets[1][t_idx]
          ]

titles = [
          r'$- \hat q \times mod$',
          r'$+ \hat q \times mod$'
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
ax.set_title(r'PRE: $D_{cont}(u,v)$', fontsize=36)

# plt.savefig(os.path.dirname(os.getcwd()) + "/Plots/ns_residual_cont.svg", format="svg",transparent=True, bbox_inches='tight')
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

plt.savefig(os.path.dirname(os.getcwd()) + "/Plots/joint_ns_mom_qhat.svg", format="svg", transparent=True, bbox_inches='tight')
plt.savefig(os.path.dirname(os.getcwd()) + "/Plots/joint_ns_mom_qhat.pdf", format="pdf", transparent=True, bbox_inches='tight')

plt.show()

# %%
#Boundary Conditions 

#PRE over the boundary conditions - Periodic
def periodic_bc_residual(u, wall='right'):
    """Apply periodic boundary conditions using edge values"""
    if wall == 'top':
        res = u[...,0, :] - u[...,-1, :]   # Top boundary equals bottom boundary
    if wall == 'bottom':
        res = u[...,-1, :]- u[...,0, :]   # Bottom boundary equals top boundary
    if wall == 'left':
        res =  u[...,:, 0] - u[...,:, -1]   # Left boundary equals right boundary
    if wall=='right':
        res = u[...,:, -1] - u[...,:, 0]   # Right boundary equals left boundary
    return res * dx

w_cal = cal_pred[:, -1].permute(0, 3, 1, 2)
w_out = pred_out[:, -1].permute(0, 3, 1, 2)
w_pred = pred_pred[:, -1].permute(0, 3, 1, 2)

w_cal_bc_residual =  periodic_bc_residual(w_cal)
w_pred_bc_residual = periodic_bc_residual(w_pred)

modulation = modulation_func(w_cal_bc_residual.numpy(), np.zeros(w_cal_bc_residual.shape))
ncf_scores = ncf_metric_joint(w_cal_bc_residual.numpy(), np.zeros(w_cal_bc_residual.shape), modulation)

#Emprical Coverage for all values of alpha to see if pred_residual lies between +- qhat. 
alpha_levels = np.arange(0.05, 0.95+0.1, 0.1)

emp_cov_res = []
for alpha in tqdm(alpha_levels):
    qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
    prediction_sets = [- qhat*modulation, + qhat*modulation]
    emp_cov_res.append(emp_cov_joint(prediction_sets, w_pred_bc_residual.numpy()))

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()
# %%
x_values = x
pred_residual = w_pred_bc_residual

alpha = 0.5
qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
prediction_sets = [- qhat*modulation,  + qhat*modulation]

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

idx = 20
t_idx = 10

plt.plot(x_values, w_pred_bc_residual[idx, t_idx], label='PRE', color='black',lw=4, ls='--', alpha=0.75)
plt.plot(x_values, prediction_sets[0][t_idx], label='Lower Joint', color='navy',lw=4, ls='--',  alpha=0.75)
plt.plot(x_values, prediction_sets[1][t_idx], label='Upper Joint', color='blue',lw=4, ls='--',  alpha=0.75)

plt.xlabel(r'$y$', fontsize=36)
plt.ylabel(r'$D_{BC}(w)$', fontsize=36)

# # Customize x-axis ticks
# plt.xticks( # 5 ticks from min to max
#     fontsize=36  # Increase font size
# )
# plt.yticks( # 5 ticks from min to max
#         np.linspace(-0.002, 0.002, 5),
#     fontsize=36  # Increase font size
# )
plt.title("Joint CP", fontsize=36)
plt.legend(fontsize=36)
plt.savefig(os.path.dirname(os.getcwd()) + "/Plots/joint_NS_BC.svg", format="svg", bbox_inches='tight')
plt.savefig(os.path.dirname(os.getcwd()) + "/Plots/joint_NS_BC.pdf", format="pdf", bbox_inches='tight')
plt.show()
# %%
'''
#NS Compressible Navier-Stokes

from Utils.ConvOps_2d import ConvOperator
#Defining the required Convolutional Operations. 
D_t = ConvOperator(domain='t', order=1)#, scale=alpha)
D_x = ConvOperator(domain='x', order=1)#, scale=beta) 
D_y = ConvOperator(domain='y', order=1)#, scale=beta)
D_x_y = ConvOperator(domain=('x', 'y'), order=1)#, scale=beta)
D_xx_yy = ConvOperator(domain=('x','y'), order=2)#, scale=gamma)

#Continuity
def residual_continuity(vars, boundary=False):
    u, v, p, rho  = vars[:,0], vars[:, 1], vars[:, 2], vars[:, 3]
    res = D_t(rho) + rho*(D_x_y(u) + D_x_y(v)) + (u+v)*(D_x_y(rho))

    if boundary:
        return res
    else: 
        return res[...,1:-1,1:-1,1:-1]
    
#Momentum 
def residual_momentum(vars, boundary=False):
    u, v, p, rho  = vars[:,0], vars[:, 1], vars[:, 2], vars[:, 3]

    res_x = rho*(D_t(u) + u*D_x(u) + v*D_y(u)) - eta*D_xx_yy(u) + D_x(p) - (zeta + eta/3) + D_x(D_x(u) + D_y(v))
    res_y = rho*(D_t(v) + u*D_x(v) + v*D_y(v))- eta*D_xx_yy(v) + D_y(p) - (zeta + eta/3) + D_y(D_x(u) + D_y(v))

    if boundary:
        return res_x + res_y
    else: 
        return res_x[...,1:-1,1:-1,1:-1] + res_y[...,1:-1,1:-1,1:-1]
    
        
'''