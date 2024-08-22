#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uncertainty quantification of a Neural PDE Surrogate solving the Wave Equation using Physics Residuals 
and guaranteed using Conformal Prediction. Prediction Selection/Rejection based on CP bounds. 

Eqn: u_tt = c**2 * (u_xx + u_yy)
c = 1.00
"""

# %%
configuration = {"Case": 'Wave',
                 "Field": 'u',
                 "Model": 'FNO',
                 "Epochs": 250,
                 "Batch Size": 50,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GeLU',
                 "Physics Normalisation": 'No',
                 "Normalisation Strategy": 'Min-Max',
                 "T_in": 1,    
                 "T_out": 60,
                 "Step": 1,
                 "Width_time": 32, 
                 "Width_vars": 0,  
                 "Modes": 8,
                 "Variables":1, 
                 "Loss Function": 'LP',
                 "UQ": 'None', #None, Dropout
                 "n_train": 800,
                 "n_test": 200,
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
from Neural_PDE.Numerical_Solvers.Wave.Wave_2D_Spectral import * 

Nx = 64 # Mesh Discretesiation 
x_min = -1.0 # Minimum value of x
x_max = 1.0 # maximum value of x
y_min = -1.0 # Minimum value of y 
y_max = 1.0 # Minimum value of y
tend = 1
c = 1.0 #Wave Speed <=1.0
t_slice = 5

sim = Wave_2D(Nx, x_min, x_max, tend, c)
dt, dx = sim.dt, sim.dx
dx , dt*t_slice
# %% 
# Utility Functions 
def gen_data(params):
    #Generating Data 
    u_sol = []
    for ii in tqdm(range(len(params))):
        x, y, t, u_sol = sim.solve()
        u_sol.append(u_soln)

    #Extraction
    u_sol = np.asarray(u_sol)[:, ::t_slice]

    #Tensorize
    u = torch.tensor(u_sol, dtype=torch.float32)
    u = u.permute(0, 2, 3, 1)#BS, Nx, Ny, Ntx_min = -1.0 # Minimum value of x
    u = u.unsqueeze(1) #Adding the variable channel

    return x, dt, u

#Generate Initial Conditions
def gen_ic(params):
    u_ic = []
    for ii in tqdm(range(len(params))):
        sim.initialise(params[ii,0], params[ii,1], params[ii,2])
        u_ic.append(sim.vv)

    u_ic = np.asarray(u_ic)
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
# Define Bounds
lb = np.asarray([10, 0.10, 0.10]) #Amplitude, x_pos, y_pos
ub = np.asarray([50, 0.50, 0.50]) #Amplitude, x_pos, y_pos

#Conv kernels --> Stencils 
from Utils.ConvOps_2d import ConvOperator
#Defining the required Convolutional Operations. 
D_tt = ConvOperator('t', 2)#, scale=alpha)
D_xx_yy = ConvOperator(('x','y'), 2)#, scale=beta)
#Additive Kernels 
D = ConvOperator()
c = torch.tensor(c, dtype=torch.float32)
D.kernel = D_tt.kernel - (c*dt/dx)**2 * D_xx_yy.kernel 
residual = D

#Loading the Model  
model = FNO_multi2d(configuration['T_in'], configuration['Step'], configuration['Modes'], configuration['Modes'], configuration['Variables'], configuration['Width_time'])
model.to(device)
print("Number of model params : " + str(model.count_params()))
model.load_state_dict(torch.load(model_loc + '/FNO_Wave_contemporary-cockatoo.pth', map_location='cpu'))

# %% 
################################################################
#Loading Calibration Data
data_loc = os.path.dirname(os.getcwd()) + '/Neural_PDE/Data'
data =  np.load(data_loc + '/Spectral_Wave_data_LHS.npz')

u_sol  = data['u'][:100]
x = data['x']
y = data['y']
t = data['t']

u = torch.tensor(u_sol, dtype=torch.float32)
u = u.permute(0, 2, 3, 1) #Adding BS and Permuting for FNO
u = u.unsqueeze(1) #Adding the variable channel

#Setting up the normalisation. 
norms = np.load(model_loc + '/FNO_Wave_contemporary-cockatoo_norms.npz')
in_normalizer, out_normalizer = normalisation(configuration['Normalisation Strategy'], norms)

cal_in, cal_out = data_loader(u, configuration['T_in'], configuration['T_out'], in_normalizer, out_normalizer, dataloader=False)
cal_pred, mse, mae = validation_AR(model, cal_in, cal_out, configuration['Step'], configuration['T_out'])
cal_out = out_normalizer.decode(cal_out)
cal_pred = out_normalizer.decode(cal_pred)

cal_residual = residual(cal_pred.permute(0,1,4,2,3)[:,0]) #Physics-Driven
# cal_residual = residual(cal_out.permute(0,1,4,2,3)[:,0]) #Data-Driven
ncf_scores = np.abs(cal_residual.numpy())

# %% 

#Plotting the fields, prediction, abs error and the residual
idx = 0
t_idx = 10
values = [cal_out[0, 0,...,t_idx],
          cal_pred[0,0,...,t_idx],
          cal_pred[0,0,...,t_idx] - cal_out[0,0,...,t_idx],
          cal_residual[idx, t_idx]
          ]

titles = [r'$u$',
          r'$\tilde u$',
          r'$u - \tilde u$',
          r'$ \frac{\partial^2 \tilde u}{\partial t^2 } - c^2 (\frac{\partial^2 \tilde u}{\partial x^2} + \frac{\partial^2 \tilde u}{\partial y^2})$'
          ]

subplots_2d(values, titles)
# %% 
#Prediction Residuals 
params = lb + (ub - lb) * lhs(3, configuration['n_pred'])
u_in_pred = gen_ic(params)
pred_pred, mse, mae = validation_AR(model, u_in_pred, torch.zeros((u_in_pred.shape[0], u_in_pred.shape[1], u_in_pred.shape[2], u_in_pred.shape[3], configuration['T_out'])), configuration['Step'], configuration['T_out'])
pred_pred = out_normalizer.decode(pred_pred)
pred_pred = pred_pred.permute(0,1,4,2,3)[:,0]
pred_residual = residual(pred_pred)

# %% 
#Selection/Rejection
alpha = 0.5
threshold = 0.5
qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
prediction_sets = [pred_residual - qhat, pred_residual + qhat]

filtered_sims = filter_sims_within_bounds(prediction_sets[0], prediction_sets[1], pred_residual.numpy(), threshold=threshold)
params_filtered = params[filtered_sims]
print(f'{len(params_filtered)} simulations rejected')

# %%
