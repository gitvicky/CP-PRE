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
                 "n_test": 100,
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
npde_loc = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(npde_loc)
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
#Estimating the residuals over the test data and the predictions.

u_out_test = test_u[...,::disc]
u_pred_test = pred_test[...,::disc]

residual_out_test = D(u_out_test.permute(0,1,3,2)[:,0])[...,1:-1, 1:-1]
residual_pred_test = D(u_pred_test.permute(0,1,3,2)[:,0])[...,1:-1, 1:-1]

# %%
