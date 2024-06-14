#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uncertainty quantification of a Neural PDE Surrogate solving the Advection Equation using Physics Residuals and guaranteed using Conformal Prediction

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
                 "n_cal": 100,
                 "n_pred": 10
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

lb = np.asarray([0.1, 0.1]) #pos, velocity
ub = np.asarray([1.0, 1.0])

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
#Inference
pred_set, mse, mae = validation_AR(model, test_a, test_u, step, T_out)

print('Testing Error (MSE) : %.3e' % (mse))
print('Testing Error (MAE) : %.3e' % (mae))

# %%
#Plotting the surrogate performance against that of the test data. 

from plot_tools import plot_2d

# idx = np.random.randint(0,ntest) 
# idx=0
# x_range = x_grid

# u_field_actual = test_u[idx, 0]
# u_field_pred = pred_set[idx, 0]

# v_min = torch.min(u_field_actual)
# v_max = torch.max(u_field_actual)


# fig = plt.figure(figsize=plt.figaspect(0.5))
# ax = fig.add_subplot(1,3,1)
# pcm = ax.plot(x_range, u_field_actual[:, 0], color='green')
# pcm = ax.plot(x_range, u_field_pred[:, 0], color='firebrick')
# ax.set_ylim([v_min, v_max])
# ax.title.set_text('t='+ str(T_in))

# ax = fig.add_subplot(1,3,2)
# pcm = ax.plot(x_range, u_field_actual[:,int(T_out/2)], color='green')
# pcm = ax.plot(x_range, u_field_pred[:, int(T_out/2)], color='firebrick')
# ax.set_ylim([v_min, v_max])
# ax.title.set_text('t='+ str(int((T_out+(T_in/2)))))
# ax.axes.yaxis.set_ticks([])

# ax = fig.add_subplot(1,3,3)
# pcm = ax.plot(x_range, u_field_actual[:, -1], color='green')
# pcm = ax.plot(x_range, u_field_pred[:, -1], color='firebrick')
# ax.title.set_text('t='+str(T_out+T_in))
# ax.set_ylim([v_min, v_max])
# ax.axes.yaxis.set_ticks([])

# %% 
# Generating the data through simulation:

n_cal = configuration['n_cal']

lb = np.asarray([0.1, 0.1]) #pos, velocity
ub = np.asarray([1.0, 1.0])

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

u = torch.tensor(u_sol, dtype=torch.float32)
u = u.unsqueeze(0).permute(0, 2, 1) #Adding BS and Permuting for FNO
u = u.unsqueeze(1) #Adding the variable channel

u_in = u[...,:configuration['T_in']]
u_out = u[...,configuration['T_in'] : configuration['T_in'] + configuration['T_out']]

# %%
#Model Predictions.
pred, mse, mae = validation_AR(model, u_in, u_out, configuration['Step'], configuration['T_out'])

# %% 
#Estimating the Residuals

# uu = pred #Prediction
uu = u_out #Solution
uu = uu.permute(0,1,3,2)
uu = uu[:,0]
uu_cal 
dx = x[-1] - x[-2]
dt = t[-1] - t[-2]

alpha = 1/dt*2
beta = 1/dx*2

# alpha, beta, gamma = 1, 1, 1

from ConvOps_1d import ConvOperator
#Defining the required Convolutional Operations. 
D_t = ConvOperator(domain='t', order=1)#, scale=alpha)
D_x = ConvOperator(domain='x', order=1)#, scale=beta) 

# Residual
residual  = D_t(uu) + (v*dt/dx) * D_x(uu) 
residual = residual[...,1:-1, 1:-1]

#Residual - Additive Kernels
D = ConvOperator()
D.kernel = D_t.kernel + (v*dt/dx) * D_x.kernel
residual = D(uu)
residual = residual[...,1:-1, 1:-1]

# %% 
from plot_tools import subplots_2d
values = [residual]
titles = ["Residual"]

subplots_2d(values, titles)

# %%
#############################################################################
#Performing the Inverse mapping from the Residuals to the Fields
#############################################################################

u_integrate = D.integrate(uu)

values=[uu[0], u_integrate[0]]
titles = ['Actual', 'Retrieved']
subplots_2d(values, titles)


from plot_tools import subplots_1d
x = x
values = {"Actual": uu[0], 
          "Retrieved": u_integrate[0]
          }
indices = [6, 12, 18, 24]
subplots_1d(x, values, indices, "Comparing Integration")

# %%
#############################################################################
# Further Comparative Studies
#############################################################################


#Comparing residuals across the prediction and comparison
u_num = u_out.permute(0,1,3,2)[:,0]
u_pred = pred.permute(0,1,3,2)[:,0]

residual_numerical = D_t(u_num) + (v*dt/dx) * D_x(u_num) 
residual_predicition =  D_t(u_pred) + (v*dt/dx) * D_x(u_pred) 


x_values = x[1:-1]
y_values = {"Numerical": residual_numerical[:, 1:-1], 
          "Prediction": residual_predicition[:, 1:-1]
          }
indices = [6, 12, 18, 24]
subplots_1d(x_values, y_values, indices, "Comparing Residauals")

# %%
#Performing the Integration by Parts without using the additive kernels

# #u = - (x.u_t + v.t.u_x) / (1+v)
# xx = torch.tensor(x).unsqueeze(0).tile(30,1)
# tt = torch.tensor(t[T_in:]).unsqueeze(0).tile(200,1).T

# u_int_parts = - (xx*D_t(uu) + v*tt*D_x(uu)) / (1+v)

u_int_parts = ( D_t.integrate(uu) + v*D_x.integrate(uu) ) / 2

values=[uu[0], u_integrate[0], u_int_parts]
titles = ['Actual', 'Retrieved', 'Ret. Parts']
subplots_2d(values, titles)

from plot_tools import subplots_1d
x_values = x
values = {"Actual": uu[0], 
          "Retrieved": u_integrate[0],
          "Ret. Parts": u_int_parts
          }
indices = [6, 12, 18, 24]
subplots_1d(x_values, values, indices, "Comparing Integrations")
# %%
