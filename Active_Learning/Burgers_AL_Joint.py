#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uncertainty quantification of a Neural PDE Surrogate solving the Wave Equation using Physics Residuals and guaranteed using Conformal Prediction
Prediction Selection/Rejection based on CP bounds / PRE / Random Sampling 
Utilised for Active Learning

Equation :  U_t + v U_x = 0
Eqn:  u_t + u*u_x =  nu*u_xx on [0,2]
Surrogate Model : FNO
Numerical Solver: https://github.com/gitvicky/Neural_PDE/tree/main/Numerical_Solvers/Burgers
"""

# %% Training Configuration - used as the config file for simvue.
configuration = {"Case": 'Burgers',
                 "Field": 'u',
                 "Model": 'FNO',
                 "Epochs": 100,
                 "Batch Size": 50,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.001,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'Tanh',
                 "Normalisation Strategy": 'Min-Max',
                 "T_in": 1,    
                 "T_out": 20,
                 "Step": 1,
                 "Width": 16, 
                 "Modes": 8,
                 "Variables":1, 
                 "Noise":0.0, 
                 "Loss Function": 'MSE',
                 "n_train": 10,
                 "n_test": 10,
                 "n_cal": 1000,
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
from Neural_PDE.Numerical_Solvers.Burgers.Burgers_1D import * 
from pyDOE import lhs

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
# %% 
#Utility Functions

def gen_data(params):
    print("Generating Data via Numerical Sims.")
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

def normalisation(norm_strategy, data=None, norms=None):
    if norm_strategy == 'Min-Max':
        normalizer = MinMax_Normalizer
    elif norm_strategy == 'Range':
        normalizer = RangeNormalizer
    elif norm_strategy == 'Gaussian':
        normalizer = GaussianNormalizer
    elif norm_strategy == 'Identity':
        normalizer = Identity

    inputs = data[..., :T_in]
    outputs = data[..., T_in:T_out+T_in]
    in_normalizer = normalizer(inputs)
    out_normalizer = normalizer(outputs)

    if data==None:
        #Loading the Normaliation values
        in_normalizer = normalizer(torch.tensor(0))
        in_normalizer.a = torch.tensor(norms['in_a'])
        in_normalizer.b = torch.tensor(norms['in_b'])
        
        out_normalizer = normalizer(torch.tensor(0))
        out_normalizer.a = torch.tensor(norms['out_a'])
        out_normalizer.b = torch.tensor(norms['out_b'])
        
    return in_normalizer, out_normalizer


#Load Simulation data into Dataloader
def data_loader(uu, in_normalizer, out_normalizer, dataloader=True, shuffle=True):

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


def train(epochs, model, train_loader, test_loader, loss_func, optimizer, scheduler):
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
    return model 
# %% 
#Define Bounds
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

################################################################
#Train Data
params = lb + (ub - lb) * lhs(3, configuration['n_train'])
x, t, u_sol_train = gen_data(params)
in_normalizer, out_normalizer = normalisation(configuration['Normalisation Strategy'], inputs= u_sol_train[..., :T_in],  outputs=u_sol_train[..., T_in:T_out+T_in])
train_loader = data_loader(u_sol_train, in_normalizer, out_normalizer)
test_loader = data_loader(u_sol_train[-10:], in_normalizer, out_normalizer) #Just kept to hefty evaluations each epoch. 

#Test Data
params = lb + (ub - lb) * lhs(3, configuration['n_test'])
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

#Training
start_time = default_timer()
model = train(epochs, model, train_loader, test_loader, loss_func, optimizer, scheduler)
train_time = default_timer() - start_time

#Evaluation
pred_test, mse, mae = validation_AR(model, test_a, test_u, step, T_out)
test_mse.append(mse)
print('Testing Error (MSE) : %.3e' % (mse))
print('Testing Error (MAE) : %.3e' % (mae))
print()

# %% 
## Using Calibration Data from smaller sample of simulations
params = lb + (ub - lb) * lhs(3, configuration['n_cal'])
x, t, u_sol = gen_data(params)
u_in_cal, u_out_cal = data_loader(u_sol, dataloader=False, shuffle=False)
u_pred_cal, mse, mae = validation_AR(model, u_in_cal, u_out_cal, step, T_out)

residual_out_cal = residual(u_out_cal.permute(0,1,3,2)[:,0])
residual_pred_cal = residual(u_pred_cal.permute(0,1,3,2)[:,0])

# %%
#Prediction Residuals 
params = lb + (ub - lb) * lhs(3, configuration['n_pred'])
u_in_pred = gen_ic(params)
pred_pred, mse, mae = validation_AR(model, u_in_pred, torch.zeros((u_in_pred.shape[0], u_in_pred.shape[1], u_in_pred.shape[2], T_out)), configuration['Step'], configuration['T_out'])
pred_pred = pred_pred.permute(0,1,3,2)[:,0]
uu_pred = pred_pred
pred_residual = residual(uu_pred)

# res = residual_out_cal #Data-Driven
res = residual_pred_cal #Physics-Driven

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
#Filtering Sims using CP. 
def filter_sims_joint(prediction_sets, y_response):
    #Identifies the simulations that falls within the bounds.
    axes = tuple(np.arange(1,len(y_response.shape)))
    return ((y_response >= prediction_sets[0]).all(axis = axes) & (y_response <= prediction_sets[1]).all(axis = axes))

# %%
#Active Learning Pipeline - 3 kinds of acquisition functions.
#  
#CP - Filtering using Joint CP over the Residual Space. 
#PRE - Filtering using the predicitons with the largest residual errors. 
#RAND - Randomly sampling from the parameter space. 

n_iterations = 5
epochs = 100
acq_func = 'CP' #CP, PRE, RAND
alpha = 0.5 #CP-alpha 

for ii in range(n_iterations):

    #Prediction Residuals 
    params = lb + (ub - lb) * lhs(3, configuration['n_pred'])
    u_in_pred = gen_ic(params)
    pred_pred, mse, mae = validation_AR(model, u_in_pred, torch.zeros((u_in_pred.shape[0], u_in_pred.shape[1], u_in_pred.shape[2], T_out)), configuration['Step'], configuration['T_out'])
    pred_pred = pred_pred.permute(0,1,3,2)[:,0]
    uu_pred = pred_pred
    pred_residual = residual(uu_pred)

    if acq_func == 'CP':
    #Selection/Rejection using Joint CP
        qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
        prediction_sets =  [- qhat*modulation, + qhat*modulation]
        filtered_sims = np.invert(filter_sims_joint(prediction_sets, pred_residual.numpy()))
        params_filtered = params[filtered_sims]
        print(f'{len(params_filtered)} predictions rejected')

    if acq_func == 'PRE':
    #Selection/Rejection using Descending order of PRE
        pred_residual_mean = torch.mean(pred_residual, axis=tuple(range(1, pred_residual.ndim)))
        pred_residual_mean_sorted, sort_index = torch.sort(pred_residual_mean)
        num_sims = int((1-alpha)*configuration['n_pred'])
        params_filtered = params[sort_index][:num_sims]
        print(f'{len(params_filtered)} predictions rejected')

    if acq_func == 'RAND':
    #Random Selection
        num_sims = int((1-alpha)*configuration['n_pred'])
        random_index = np.random.randint(0, configuration['n_pred'], num_sims)
        params_filtered = params[random_index]
        print(f'{len(params_filtered)} predictions selected')


    #Numerical Sim over the Rejected Predictions and adding to training data.
    x, t, u_sol = gen_data(params_filtered)
    u_sol_train = torch.vstack((u_sol_train, u_sol))
    train_loader = data_loader(u_sol_train)

    #Fine-tuning with the Sampled Sims. 
    start_time = default_timer()
    model = train(epochs, model, train_loader, test_loader, loss_func, optimizer, scheduler)
    train_time = default_timer() - start_time

    #Evaluation
    pred_test, mse, mae = validation_AR(model, test_a, test_u, step, T_out)
    test_mse.append(mse)
    print('Testing Error (MSE) : %.3e' % (mse))
    print('Testing Error (MAE) : %.3e' % (mae))
    print()

# %% 
plt.plot(test_mse)
plt.ylabel('MSE')
plt.xlabel('AL - Iterations')
plt.title(acq_func)
# %%
# mse_cp = []
