#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
28th Feb, 2023

Learning the required Convolutional Kernels using LBFGS 
    1. Try and learn the finite difference stencil for performing the residual estimation
    2. If we are able to retrieve the FD stencils, attempt tp learnt the TranspConv kernel that maps from the residual to the field. 

Using the 1D Laplacian for this exploration. 
"""

# %% 
import os
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from pyDOE import lhs
from tqdm import tqdm 
from sympy import *
# init_printing(use_unicode=True)

# seed = np.random.randint(1000)
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

#Setting up CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% 
#Generating the datasets: Performing the Laplace Operator over a range of ICs. 
#This will be the training input
def generate_IC(N):

    x = np.linspace(1, -1, 31)
    # y = x.copy()
    # xx, yy = np.meshgrid(x, y)

    lb = np.asarray([-10, 0.10, 0.10]) #Amp, x-pos, y-pos 
    ub = np.asarray([-50, 0.70, 0.70]) #Amp, x-pos, y-pos
    
    params = lb + (ub-lb)*lhs(3, N)
    amp, x_pos, y_pos = params[:, 0], params[:, 1], params[:, 2]
    
    u_ic = []
    for ii in range(N):
        u_ic.append(np.exp(amp[ii]*((x-x_pos[ii])**2 )))
    return np.asarray(u_ic)

N_samples = 10000
uu = generate_IC(N_samples)
uu = torch.Tensor(uu)

# %% 
#Obtaining the laplacian of the field evaluated using the 5-point stencil. 
#This will be the training target
def laplace_stencil(X):    
    laplace_kernel = torch.tensor([[1., -2., 1.]])
    
    conv = F.conv1d(X.view(X.shape[0], 1, X.shape[1]), laplace_kernel.view(1,1,3))
    return conv

uu_laplace = laplace_stencil(uu)

# %% 
#Setting up the one layer convolutional network 

# 1D convolutional laplace 
class conv_laplace(nn.Module):
    def __init__(self, activation=None):
        super(conv_laplace, self).__init__()

        #Convolutional Layers
        self.conv = nn.Conv1d(1, 1, kernel_size=(3), stride=1)

    def forward(self, x):
        x = self.conv(x)
        return x
         

# %% 
#Setting up simvue 
from simvue import Run

configuration = {"Case": 'Forward',
                 "Epochs": 10,
                 "Batch Size": 10000,
                 "Optimizer": 'LBFGS',
                #  "Optimizer": 'Adam',
                #  "Learning Rate": 1e-3,
                #  "Scheduler Step": 100,
                #  "Scheduler Gamma": 0.5,
}

run = Run(mode='disabled')
run.init(folder="/Residuals_UQ/stencil_inversion", tags=['Forward Kernel', configuration['Optimizer'], 'FD', '1D'], metadata=configuration)

learnt_laplace = conv_laplace().to(device)

loss_func = torch.nn.MSELoss()
if configuration['Optimizer'] == 'LBFGS':
    optimizer = torch.optim.LBFGS(learnt_laplace.parameters(), history_size=10, max_iter=4)
elif configuration['Optimizer'] == 'Adam':
    optimizer = torch.optim.Adam(learnt_laplace.parameters(), lr=configuration['Learning Rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
# %% 
#Prepping the data. 
X_train = uu.view(uu.shape[0], 1, uu.shape[1])
Y_train = uu_laplace

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), batch_size=configuration['Batch Size'], shuffle=True)

# %% 
#Defining the Closure required to evaluaet LBFGS
def closure():
    optimizer.zero_grad()
    y_out = learnt_laplace(xx)
    loss = loss_func(y_out, yy)
    loss.backward()
    return loss

#Training the Convolutional Network 
epochs = configuration['Epochs']
loss_val = []
for ii in tqdm(range(epochs)):    
    for xx, yy in train_loader:
        xx, yy = xx.to(device), yy.to(device)

        if configuration['Optimizer'] == 'LBFGS':
            loss = closure()        
            optimizer.step(closure)

        elif configuration['Optimizer'] == 'Adam':
            optimizer.zero_grad()
            y_out = learnt_laplace(xx)
            loss = loss_func(y_out, yy)
            loss.backward()
            optimizer.step()
            scheduler.step()
    run.log_metrics({'Train Loss': loss.item()})

#Getting an output to plot 
y_out = learnt_laplace(xx)

# run.save(learnt_laplace.conv.weight.detach().cpu().numpy(), 'output', name='learnt_forward_stencil.npy')

# %% 
#Printing the Convolution Operation. 

u1, u2, u3, u4 = Symbol("u1"), Symbol("u2"), Symbol("u3"), Symbol("u4")


field_matrix = Matrix([[u1, u2, u3, u4]])
forward_kernel = Matrix(learnt_laplace.conv.weight.detach().numpy()[0,0])

convolution = Matrix([Matrix(field_matrix[:-1]).T*forward_kernel, Matrix(field_matrix[1:]).T*forward_kernel])

print('Convolved Output')
print(convolution)

#Saving the Code
run.save(os.path.abspath(__file__), 'code')

#Closing the run. 
run.close()

# %%
#Learning the inverse of the laplace 

# 1D tranposed convolution for the inverse laplace
class transp_conv_inv_laplace(nn.Module):
    def __init__(self, activation=None):
        super(transp_conv_inv_laplace, self).__init__()

        #Convolutional Layers
        self.transp_conv = nn.ConvTranspose1d(1, 1, (3), stride=1)

    def forward(self, x):
        x =self.transp_conv(x)
        return x
    

inv_laplace = transp_conv_inv_laplace().to(device)

configuration = {"Case": 'Inverse',
                 "Epochs": 50,
                 "Batch Size": 10000,
                 "Optimizer": 'LBFGS',
                #  "Optimizer": 'Adam',
                #  "Learning Rate": 5e-3,
                #  "Scheduler Step": 1000,
                #  "Scheduler Gamma": 0.5,
}
run = Run(mode='disabled')
run.init(folder="/Residuals_UQ/stencil_inversion", tags=['Inverse Kernel', configuration['Optimizer'], 'FD', '1D'], metadata=configuration)

loss_func = torch.nn.MSELoss()

if configuration['Optimizer'] == 'LBFGS':
    optimizer = torch.optim.LBFGS(inv_laplace.parameters(), history_size=10, max_iter=4)
elif configuration['Optimizer'] == 'Adam':
    optimizer = torch.optim.Adam(inv_laplace.parameters(), lr=configuration['Learning Rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])

# %% 
#Prepping the data. 
X_train = uu_laplace
Y_train = uu.view(uu.shape[0], 1, uu.shape[1])

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), batch_size=configuration['Batch Size'], shuffle=True)

# %% 

#Defining the Closure required to evaluaet LBFGS
def closure():
    optimizer.zero_grad()
    y_out = inv_laplace(xx)
    loss = loss_func(y_out, yy)
    loss.backward()
    return loss

#Training the Transposed Convolutional Network 
epochs = configuration['Epochs']
loss_val = []
for ii in tqdm(range(epochs)):    
    for xx, yy in train_loader:
        xx, yy = xx.to(device), yy.to(device)
        if configuration['Optimizer'] == 'LBFGS':
            loss = closure()        
            optimizer.step(closure)
        
        elif configuration['Optimizer'] == 'Adam':
            optimizer.zero_grad()
            y_out = inv_laplace(xx)
            loss = loss_func(y_out, yy)
            loss.backward()
            optimizer.step()
            scheduler.step()
    run.log_metrics({'Train Loss': loss.item()})

#Getting an output to plot 
y_out = inv_laplace(xx)

run.save(inv_laplace.transp_conv.weight.detach().cpu().numpy(), 'output', name='learnt_inverse_stencil.npy')

# %%
#Printing the Convolution Operation. 


field_matrix = Matrix([[u1, u2, u3, u4]])
inverse_kernel = Matrix(inv_laplace.conv.weight.detach().numpy()[0,0])



deconvolution = []

print('Convolved Output')
print(convolution)
#Saving the Code
run.save(os.path.abspath(__file__), 'code')

#Closing the simvue run. 
run.close()
# %%
