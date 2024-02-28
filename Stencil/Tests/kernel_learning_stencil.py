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
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from pyDOE import lhs
from tqdm import tqdm 
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

N = 10000
uu = generate_IC(N)
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
    

learnt_laplace = conv_laplace()

loss_func = torch.nn.MSELoss() #LogitsLoss contains the sigmoid layer - provides numerical stability. 
optimizer = torch.optim.Adam(learnt_laplace.parameters(), lr=1e-3)

# %% 
#Prepping the data. 
X_train = uu.view(uu.shape[0], 1, uu.shape[1], uu.shape[2])
Y_train = uu_laplace

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), batch_size=100, shuffle=True)


# %% 
#Training the Convolutional Network 
epochs = 1000
loss_val = []
for ii in tqdm(range(epochs)):    
    for xx, yy in train_loader:
        optimizer.zero_grad()
        y_out = learnt_laplace(xx)
        loss = loss_func(y_out, yy)
        loss.backward()
        optimizer.step()
    loss_val.append(loss.item())

torch.save(learnt_laplace, "learnt_laplace.pth")
plt.plot(np.arange(1, epochs+1), np.asarray(loss_val))
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training Loss")

print(learnt_laplace.conv.weight)

# %%
#Plotting and comparing 

fig = plt.figure(figsize=(10, 5))
# plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.1)
plt.title('FD stencil')
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

loss_func = torch.nn.MSELoss() #LogitsLoss contains the sigmoid layer - provides numerical stability. 
optimizer = torch.optim.Adam(inv_laplace.parameters(), lr=1e-3)

# %% 
#Prepping the data. 
X_train = uu_laplace
Y_train = uu.view(uu.shape[0], 1, uu.shape[1], uu.shape[2])[:,:,1:-1,1:-1]

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), batch_size=1000, shuffle=True)


# %% 
#Training the Convolutional Network 
epochs = 1000
loss_val = []
for ii in tqdm(range(epochs)):    
    for xx, yy in train_loader:
        optimizer.zero_grad()
        y_out = inv_laplace(xx)[:, :, 1:-1, 1:-1]
        loss = loss_func(y_out, yy)
        loss.backward()
        optimizer.step()
    loss_val.append(loss.item())

torch.save(inv_laplace, "inv_laplace.pth")
plt.plot(np.arange(1, epochs+1), np.asarray(loss_val))
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training Loss")

# %%
fig = plt.figure(figsize=(10, 5))
# plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.1)
plt.title('Inverse FD stencil')
ax = fig.add_subplot(1,2,1)
pcm =ax.imshow(yy[-1,0][:, :, 1:-1, 1:-1], cmap=cm.coolwarm)
ax.title.set_text('Actual Field')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(1,2,2)
pcm =ax.imshow(y_out[-1,0].detach().numpy(), cmap=cm.coolwarm)
ax.title.set_text('Retrieved Field')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

# %%
