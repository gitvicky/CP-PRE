#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
28th Feb, 2023

Learning the required Convolutional Kernels 
    1. Try and learn the finite difference stencil for performing the residual estimation
    2. If we are able to retrieve the FD stencils, attempt tp learnt the TranspConv kernel that maps from the residual to the field. 

Using the wave equation as the base test case for this experiment.
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

seed = np.random.randint(1e6)
torch.manual_seed(seed)
np.random.seed(seed)

#Setting up CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% 
#Generating the datasets: Performing the Laplace Operator over a range of ICs. 
#This will be the training input
def generate_IC(N):

    x = np.linspace(1, -1, 31)
    y = x.copy()
    xx, yy = np.meshgrid(x, y)

    lb = np.asarray([-10, 0.10, 0.10]) #Amp, x-pos, y-pos 
    ub = np.asarray([-50, 0.70, 0.70]) #Amp, x-pos, y-pos
    
    
    params = lb + (ub-lb)*lhs(3, N)
    amp, x_pos, y_pos = params[:, 0], params[:, 1], params[:, 2]
    
    u_ic = []
    for ii in range(N):
        u_ic.append(np.exp(amp[ii]*((xx-x_pos[ii])**2 + (yy-y_pos[ii])**2)))
    return np.asarray(u_ic)

N_samples = 10000
uu = generate_IC(N_samples)
uu = torch.Tensor(uu)

# %% 
#Obtaining the laplacian of the field evaluated using the 5-point stencil. 
#This will be the training target
def laplace_stencil(X):    
    laplace_kernel = torch.tensor([[0., 1., 0.],
                                   [1., -4., 1.],
                                   [0, 1., 0.]])
    
    conv = F.conv2d(X.view(X.shape[0], 1, X.shape[1], X.shape[2]), laplace_kernel.view(1,1,3,3))
    return conv

uu_laplace = laplace_stencil(uu)

# %% 
#Setting up the one layer convolutional network 

# 1D convolutional laplace 
class conv_laplace(nn.Module):
    def __init__(self, activation=None):
        super(conv_laplace, self).__init__()

        #Convolutional Layers
        self.conv = nn.Conv2d(1, 1, (3,3), stride=1)

    def forward(self, x):
        x = self.conv(x)
        return x
    

# %% 
#Setting up simvue 
from simvue import Run

configuration = {"Case": 'Forward',
                 "Epochs": 1000,
                 "Batch Size": 100,
                 "Optimizer": 'Adam',
                 "Learning Rate": 1e-3,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
}

run = Run()
run.init(folder="/Residuals_UQ/stencil_inversion", tags=['Forward Kernel', 'Convolutional Kernel', 'FD Stencil', 'Wave'], metadata=configuration)

learnt_laplace = conv_laplace()

loss_func = torch.nn.MSELoss() #LogitsLoss contains the sigmoid layer - provides numerical stability. 
optimizer = torch.optim.Adam(learnt_laplace.parameters(), lr=configuration['Learning Rate'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
# %% 
#Prepping the data. 
X_train = uu.view(uu.shape[0], 1, uu.shape[1], uu.shape[2])
Y_train = uu_laplace

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), batch_size=100, shuffle=True)

# %% 
#Training the Convolutional Network 
epochs = configuration['Epochs']
loss_val = []
for ii in tqdm(range(epochs)):    
    for xx, yy in train_loader:
        optimizer.zero_grad()
        y_out = learnt_laplace(xx)
        loss = loss_func(y_out, yy)
        loss.backward()
        optimizer.step()
        scheduler.step()
    run.log_metrics({'Train Loss': loss.item()})

run.save(learnt_laplace.conv.weight.detach().numpy(), 'output', name='learnt_forward_stencil.npy')

# %%
#Plotting and comparing 

fig = plt.figure(figsize=(10, 5))

mini = torch.min(yy[-1,0])
maxi = torch.max(yy[-1,0])

# plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.1)
plt.title('FD stencil')

# Selecting the axis-X making the bottom and top axes False. 
plt.tick_params(axis='x', which='both', bottom=False, 
                top=False, labelbottom=False) 
  
# Selecting the axis-Y making the right and left axes False 
plt.tick_params(axis='y', which='both', right=False, 
                left=False, labelleft=False) 
  # Remove frame
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)


ax = fig.add_subplot(1,2,1)
pcm =ax.imshow(yy[-1,0], cmap=cm.coolwarm)
ax.title.set_text('Actual Kernel')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(1,2,2)
pcm =ax.imshow(y_out[-1,0].detach().numpy(), cmap=cm.coolwarm)
ax.title.set_text('Learnt Kernel')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

#Saving the images
# run.save(fig, 'output', name='FD_Stencils.png')

#Saving the Code
run.save(os.path.abspath(__file__), 'code')

#Closing the run. 
run.close()

# %%
# %% 
#Learning the inverse of the laplace 

# 1D tranposed convolution for the inverse laplace
class transp_conv_inv_laplace(nn.Module):
    def __init__(self, activation=None):
        super(transp_conv_inv_laplace, self).__init__()

        #Convolutional Layers
        self.transp_conv = nn.ConvTranspose2d(1, 1, (3,3), stride=1)

    def forward(self, x):
        x =self.transp_conv(x)
        return x
    

inv_laplace = transp_conv_inv_laplace()

configuration = {"Case": 'Inverse',
                 "Epochs": 5000,
                 "Batch Size": 100,
                 "Optimizer": 'Adam',
                 "Learning Rate": 5e-3,
                 "Scheduler Step": 1000,
                 "Scheduler Gamma": 0.5,
}
run = Run()
run.init(folder="/Residuals_UQ/stencil_inversion", tags=['Inverse Kernel', 'Convolutional Kernel', 'FD Stencil','Wave'], metadata=configuration)

loss_func = torch.nn.MSELoss() #LogitsLoss contains the sigmoid layer - provides numerical stability. 
optimizer = torch.optim.Adam(inv_laplace.parameters(), lr=configuration['Learning Rate'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
# %% 
#Prepping the data. 
X_train = uu_laplace
Y_train = uu.view(uu.shape[0], 1, uu.shape[1], uu.shape[2])[:,:,1:-1,1:-1]

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), batch_size=1000, shuffle=True)

# %% 
#Training the Transposed Convolutional Network 
epochs = configuration['Epochs']
loss_val = []
for ii in tqdm(range(epochs)):    
    for xx, yy in train_loader:
        optimizer.zero_grad()
        y_out = inv_laplace(xx)[:, :, 1:-1, 1:-1]
        loss = loss_func(y_out, yy)
        loss.backward()
        optimizer.step()
        scheduler.step()
    run.log_metrics({'Train Loss': loss.item()})

run.save(inv_laplace.transp_conv.weight.detach().numpy(), 'output', name='learnt_inverse_stencil.npy')


# %%
fig = plt.figure(figsize=(10, 5))

mini = torch.min(yy[-1,0][1:-1, 1:-1])
maxi = torch.max(yy[-1,0][1:-1, 1:-1])


# plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.1)
plt.title('Inverse FD stencil')


# Selecting the axis-X making the bottom and top axes False. 
plt.tick_params(axis='x', which='both', bottom=False, 
                top=False, labelbottom=False) 
  
# Selecting the axis-Y making the right and left axes False 
plt.tick_params(axis='y', which='both', right=False, 
                left=False, labelleft=False) 
  # Remove frame
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)


ax = fig.add_subplot(1,2,1)
pcm =ax.imshow(yy[-1,0][1:-1, 1:-1], cmap=cm.coolwarm, vmin=mini, vmax=maxi)
ax.title.set_text('Actual Field')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(1,2,2)
pcm =ax.imshow(y_out[-1,0].detach().numpy(), cmap=cm.coolwarm,  vmin=mini, vmax=maxi)
ax.title.set_text('Retrieved Field')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

#Saving the Image 
# run.save(fig, 'output', name='Inverse_Stencils.png')

#Saving the Code
run.save(os.path.abspath(__file__), 'code')

#Closing the simvue run. 
run.close()
# %%
