#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
26th March, 2024

Tests to determine the position and axis of the stencils with respect to the spatio-temporal grids
Deploying the Scheme on the 2D heat equation with an analytical solution. 
"""
# %% 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
def heat_2d_analytical(x, y, t, alpha):
    return np.sin(x/t) + np.cos(y/t)
    return 1 / (4 * np.pi * alpha * t) * np.exp(-(x**2 + y**2) / (4 * alpha * t))

# Parameters
alpha = 0.1  # Thermal diffusivity
t_values = [0.1, 0.5, 1.0]  # Time points

# Create a grid of points
grid_res = 500
x = np.linspace(-1, 1, grid_res)
y = np.linspace(-1, 1, grid_res)
X, Y = np.meshgrid(x, y)

# Plot the solution at different time points
fig, axs = plt.subplots(1, len(t_values), figsize=(12, 4))

for i, t in enumerate(t_values):
    Z = heat_2d_analytical(X, Y, t, alpha)
    cs = axs[i].contourf(X, Y, Z, cmap='hot')
    axs[i].set_title(f't = {t}')
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('y')
    axs[i].set_aspect('equal')
    
    # Add colorbar to each subplot
    cbar = fig.colorbar(cs, ax=axs[i])
    cbar.set_label('Temperature')
plt.tight_layout()
plt.show()
# %%
#Taking the analytical derivatives across the 2d-time domains. 

def func_u_t(x,y,t,alpha):
    return np.exp(-(x**2 + y**2)/4*alpha*t) * (-4*alpha*t + x**2 + y**2) * (1/(16*np.pi*alpha**2*t**3))

def func_u_xx(x,y,t,alpha):
    return (x**2 - 2*alpha*t) * np.exp(-(x**2 + y**2)/4*alpha*t) * (1/(16*np.pi*alpha**3*t**3))

def func_u_yy(x,y,t,alpha):
    return (y**2 - 2*alpha*t) * np.exp(-(x**2 + y**2)/4*alpha*t) * (1/(16*np.pi*alpha**3*t**3))

# %% 
dx = x[-1] - x[-2]
dt = 0.001
t_range = np.arange(0.1,0.2+dt,dt)
t_res = len(t_range)

u = []
u_t = []
u_xx = []
u_yy = []

for i, t in tqdm(enumerate(t_range)):
    u.append(heat_2d_analytical(X,Y,t,alpha))
    u_t.append(func_u_t(X,Y,t,alpha))
    u_xx.append(func_u_xx(X,Y,t,alpha))
    u_yy.append(func_u_yy(X,Y,t,alpha))

u = np.asarray(u)
u_t = np.asarray(u_t)
u_xx = np.asarray(u_xx)
u_yy = np.asarray(u_yy)
residual = u_t - alpha*(u_xx + u_yy)

# %%
#Taking the derivatives using the Finite Difference Scheme with the Convolutional Kernels. 
import torch 
from torch.nn import functional as F
#Some tests to ensure that the kernels are aligned to the correct axis. 
#Compaing the Laplacian Operator implemented in 2D and 3D 

alpha = 1/dx**2
beta = 1/(2*dt)
# t_res = len(t_range[-10:])
three_p_stencil = alpha * torch.tensor([[0, 1, 0],
                           [0, -2, 0],
                           [0, 1, 0]], dtype=torch.float32)

five_p_stencil = alpha * torch.tensor([[0., 1., 0.],
                       [1., -4., 1.],
                       [0., 1., 0.]])

nine_p_stencil = alpha * torch.tensor([[0, 0, -1/12, 0, 0],
                                       [0, 0, 4/3, 0, 0],
                                       [-1/12, 4/3, -5/2, 4/3, -1/12],
                                       [0, 0, 4/3, 0, 0],
                                       [0, 0, -1/12, 0, 0]]
                                       )

two_p_stencil = beta * torch.tensor([[0, -1, 0],
                           [0, 0, 0],
                           [0, 1, 0]], dtype=torch.float32)
                           
u_tensor = torch.tensor(u, dtype=torch.float32)

u_xx_conv_2d =  F.conv2d(u_tensor[-1].view(1,1,grid_res,grid_res), three_p_stencil.view(1,1,3,3)/dx**2,padding=1)
u_yy_conv_2d =  F.conv2d(u_tensor[-1].view(1,1,grid_res,grid_res), three_p_stencil.T.view(1,1,3,3)/dx**2,padding=1)
u_xx_yy_conv_2d = F.conv2d(u_tensor[-1].view(1,1,grid_res,grid_res), five_p_stencil.view(1,1,3,3)/dx**2,padding=1)

stencil_xx_yy = torch.zeros(3,3,3)
stencil_xx_yy[: ,:, 1] = five_p_stencil
u_xx_yy_conv_3d = F.conv3d(u_tensor.view(1,1,t_res,grid_res,grid_res), stencil_xx_yy.view(1,1,3,3,3),padding=1)

# stencil_xx_yy = torch.zeros(5,5,5)
# stencil_xx_yy[: ,:, 1] = nine_p_stencil
# u_xx_yy_conv_3d = F.conv3d(u_tensor.view(1,1,t_res,grid_res,grid_res), stencil_xx_yy.view(1,1,5,5,5),padding=2)

stencil_t = torch.zeros(3,3,3)
stencil_t[: ,:, 1] = two_p_stencil
u_t_conv_3d = F.conv3d(u_tensor.view(1,1,t_res,grid_res,grid_res), stencil_t.view(1,1,3,3,3),padding=1)

# %%
#Plotting the input, laplace, inverse 
alpha = 0.1
from matplotlib import pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
fig = plt.figure(figsize=(12, 8))

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


ax = fig.add_subplot(2,2,1)
pcm =ax.imshow(alpha*u_xx_yy_conv_3d[0,0,-1], cmap=cm.coolwarm)#, vmin=mini, vmax=maxi)
ax.title.set_text(r'$\alpha * (u_{xx} + u_{yy})$')
ax.set_xlabel('x')
ax.set_ylabel('CK as FDS')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(2,2,2)
pcm =ax.imshow(u_t_conv_3d[0,0,-1], cmap=cm.coolwarm)#,  vmin=mini, vmax=maxi)
ax.title.set_text(r'$u_t$')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(2,2,3)
pcm =ax.imshow(alpha*(u_xx[-1] + u_yy[-1]), cmap=cm.coolwarm)#, vmin=mini, vmax=maxi)
ax.title.set_text(r'$\alpha * (u_{xx} + u_{yy})$')
ax.set_xlabel('x')
ax.set_ylabel('Analytical')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(2,2,4)
pcm =ax.imshow(u_t[-1], cmap=cm.coolwarm)#,  vmin=mini, vmax=maxi)
ax.title.set_text(r'$u_t$')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

# %%
