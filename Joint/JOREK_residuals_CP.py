#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reduced MHD with Multi-Blob Diffusion utilised in https://iopscience.iop.org/article/10.1088/1741-4326/ad313a

In toroidal coordinates - x: R, y: Z

"""

# %%
configuration = {"Case": 'MHD',
                 "Field": 'rho, Phi, T',
                 "Model": 'FNO',
                 "Epochs": 500,
                 "Batch Size": 10,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GeLU',
                 "Physics Normalisation": 'No',
                 "Normalisation Strategy": 'Min-Max',
                 "T_in": 10,    
                 "T_out": 40,
                 "Step": 5,
                 "Width_time": 32, 
                 "Width_vars": 0,  
                 "Modes": 16,
                 "Variables":3, 
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

# %% 
#Setting up locations. 
file_loc = os.getcwd()
# data_loc = file_loc + '/Neural_PDE/Data'
data_loc = '/Users/Vicky/Documents/UKAEA/Data/Multi-Blobs'
model_loc = file_loc + '/Weights'
plot_loc = file_loc + '/Plots'
#Setting up the seeds and devices
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

#Setting up the Utility Functions 
def stacked_fields(variables):
    stack = []
    for var in variables:
        var = torch.from_numpy(var) #Converting to Torch
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

# %% 
# Loading the Calibration Data
t1 = default_timer()
data = data_loc + '/FNO_MHD_data_multi_blob_2000_T50.npz' #2000 simulation dataset

field = configuration['Field']
field = ['rho', 'Phi', 'T']
num_vars = configuration['Variables']

rho = np.load(data)['rho'].astype(np.float32) / 1e20
phi = np.load(data)['Phi'].astype(np.float32) / 1e5
T = np.load(data)['T'].astype(np.float32) / 1e6

rho = np.nan_to_num(rho)
phi = np.nan_to_num(phi)
T = np.nan_to_num(T)

x_grid = np.load(data)['Rgrid'][0, :].astype(np.float32)
y_grid = np.load(data)['Zgrid'][:, 0].astype(np.float32)
t_grid = np.load(data)['time'].astype(np.float32)


x_slice = 1 
vars = stacked_fields([rho, phi, T])[:, :, ::x_slice, ::x_slice, :]
vars = np.delete(vars, (11, 160, 222, 273, 303, 357, 620, 797, 983, 1275, 1391, 1458, 1554, 1600, 1613, 1888, 1937, 1946, 1959), axis=0) #2000 dataset

field = ['rho', 'phi', 'T']

n_cal = configuration['n_cal']
u_in_cal = vars[:n_cal,...,:configuration['T_in']]
u_out_cal = vars[:n_cal,...,configuration['T_in'] : configuration['T_in'] + configuration['T_out']]

# %% 
#Normalisation
norms = np.load(model_loc + '/FNO_MHD_grating-cookie_norms.npz')
#Loading the Normaliation values
in_normalizer = MinMax_Normalizer(u_in_cal)
in_normalizer.a = torch.tensor(norms['in_a'])
in_normalizer.b = torch.tensor(norms['in_b'])

out_normalizer = MinMax_Normalizer(u_out_cal)
out_normalizer.a = torch.tensor(norms['out_a'])
out_normalizer.b = torch.tensor(norms['out_b'])

u_in_cal = in_normalizer.encode(u_in_cal)
u_out_encoded_cal = out_normalizer.encode(u_out_cal)

# %%
#Load the model and Obtain predictions
model = FNO_multi2d(configuration['T_in'], configuration['Step'], configuration['Modes'], configuration['Modes'], configuration['Variables'], configuration['Width_time'])
model.load_state_dict(torch.load(model_loc + '/FNO_MHD_grating-cookie.pth', map_location='cpu'))

#Model Predictions.
pred_encoded_cal, mse, mae = validation_AR(model, u_in_cal, u_out_encoded_cal, configuration['Step'], configuration['T_out'])

print('Calibration (MSE) Error: %.3e' % (mse))
print('Calibration (MAE) Error: %.3e' % (mae))

#Denormalising the predictions
pred_cal = out_normalizer.decode(pred_encoded_cal.to(device)).cpu()

# %% 
#Residual Functions

dx = torch.tensor(np.asarray(x_grid[-1] - x_grid[-2]), dtype=torch.float32)
dy = dx
dt = torch.tensor(np.asarray(t_grid[-1] - t_grid[-2]), dtype=torch.float32)

alpha = 1/dt*2
beta = 1/dx*2
gamma = 1/dx**2                 

alpha, beta, gamma = 1,1,1

R = torch.tensor(x_grid, dtype=torch.float32)
Z = torch.tensor(y_grid, dtype=torch.float32)

#Coefficients 
D = torch.tensor(3.4, dtype=torch.float32)
mu = torch.tensor(2.25 * 1e-6, dtype=torch.float32)
K = torch.tensor(2.25 * 1e-7, dtype=torch.float32)
gamma =  torch.tensor(5/3, dtype=torch.float32)

from Utils.ConvOps_2d import ConvOperator
#Defining the required Convolutional Operations. 
D_t = ConvOperator(domain='t', order=1, scale=alpha)
D_R = ConvOperator(domain='x', order=1, scale=beta) 
D_Z = ConvOperator(domain='y', order=1, scale=beta)
D_RR = ConvOperator(domain='x', order=2, scale=gamma)
D_ZZ = ConvOperator(domain='y', order=2, scale=gamma)

#Continuity 
def residual_continuity(vars, boundary=False, norms=False):
    rho, phi, T = unstack_fields(vars, axis=1, variable_names=field)#Prediction 

    if norms:
        res = 2*dx*dy*D_t(rho) - (dt)*R*(D_R(rho)*D_Z(phi) - D_R(phi)*D_Z(rho)) - (2*dt*dy)*2*rho*D_Z(phi) - (4*dt)*D*(D_RR(rho) + (1/R)*D_R(rho) + D_ZZ(rho))
    else:
        res = D_t(rho) - R*(D_R(rho)*D_Z(phi) - D_R(phi)*D_Z(rho)) - 2*rho*D_Z(phi) - D*(D_RR(rho) + (1/R)*D_R(rho) + D_ZZ(rho))
    
    if boundary:
        return res
    else: 
        return res[...,1:-1,1:-1,1:-1]
    
#Temperature
def residual_temperature(vars, boundary=False, norms=False):
    rho, phi, T = unstack_fields(vars, axis=1, variable_names=field)#Prediction 

    if norms:
        raise Exception("Norm not implemented yet")    
    
    else:
        res = T*D_t(rho) + rho*D_t(T) - rho*R*(D_R(T)*D_Z(phi) - D_R(phi)*D_Z(T)) +  \
            T*R*(D_R(rho)*D_Z(phi) - D_R(phi)*D_Z(rho)) + \
            2*gamma*rho*T*D_Z(phi) + \
            K * (D_RR(T) + (1/R)*D_R(T) + D_ZZ(T))    
    
    if boundary:
        return res
    else: 
        return res[...,1:-1,1:-1,1:-1]

#Continuity
cal_out_residual = residual_continuity(u_out_cal)
cal_pred_residual = residual_continuity(pred_cal)

# #Temperature
# cal_out_residual = residual_temperature(u_out_cal)
# cal_pred_residual = residual_temperature(pred_cal)

modulation = modulation_func(cal_out_residual.numpy(), cal_pred_residual.numpy())
ncf_scores = ncf_metric_joint(cal_out_residual.numpy(), cal_pred_residual.numpy(), modulation)

# %% 
#Obtaining the predictions 
n_pred = configuration['n_pred']
u_in_pred = vars[n_cal:n_cal+n_pred,...,:configuration['T_in']]
u_out_pred = vars[n_cal:n_cal+n_pred,...,configuration['T_in'] : configuration['T_in'] + configuration['T_out']]

#Normalisations
u_in_pred = in_normalizer.encode(u_in_pred)
u_out_encoded_pred = out_normalizer.encode(u_out_pred)

#Model Predictions.
pred_encoded_pred, mse, mae = validation_AR(model, u_in_pred, u_out_encoded_pred, configuration['Step'], configuration['T_out'])

print('Prediction (MSE) Error: %.3e' % (mse))
print('Prediction (MAE) Error: %.3e' % (mae))

#Denormalising the predictions
pred_pred = out_normalizer.decode(pred_encoded_pred.to(device)).cpu()

#Continuity
pred_out_residual = residual_continuity(u_out_pred)
pred_pred_residual = residual_continuity(pred_pred)

# #Temperature
# pred_out_residual = residual_temperature(u_out_pred)
# pred_pred_residual = residual_temperature(pred_pred)


#Emprical Coverage for all values of alpha 
alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov_res = []
for alpha in tqdm(alpha_levels):
    qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
    prediction_sets = [pred_pred_residual.numpy() - qhat*modulation, pred_pred_residual.numpy() + qhat*modulation]
    emp_cov_res.append(emp_cov_joint(prediction_sets, pred_out_residual.numpy()))

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()
# %% 
# using PRE only 
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
    emp_cov_res.append(emp_cov_joint(prediction_sets, pred_pred_residual.numpy()))

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()

# %% 