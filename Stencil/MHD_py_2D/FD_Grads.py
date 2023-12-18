#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

---------------------------------------------------------------------------------------
Testing out the FD gradient estimation of spatio-temporal fields for a 2D Constrained transport MHD scenario
Demonstrating the Stencil Approach. 

MHD Equation solved using the finite volume method - gradient estimation using FD stencils over the primitive form of MHD equations. 
------------------------------------------------------

---------------------------------------------------------------------------------------
"""

# %% 
#Importing the necessary packages 
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm_gui
import sympy 
import torch 
# %% 
#Obtaining the numerical solution of the 2D Navier-Stokes incompressible flow 

from ConstrainedMHD_numerical import *

N = 128
boxsize = 1.0
tEnd = 0.5
rho, u, v, p, bx, by, dt = solve(N, boxsize, tEnd)
dx = boxsize/N

#Ignoring the varying dt for the time being 
dt = np.mean(dt)
#%%
#Utilising the stencil by way of convolutions using pytorch 
import torch.nn.functional as F

rho_tensor =torch.tensor(rho, dtype=torch.float32).reshape(1,1, rho.shape[0], rho.shape[1], rho.shape[2])
u_tensor =torch.tensor(u, dtype=torch.float32).reshape(1,1, u.shape[0], u.shape[1], u.shape[2])
v_tensor =torch.tensor(v, dtype=torch.float32).reshape(1,1, v.shape[0], v.shape[1], v.shape[2])
p_tensor =torch.tensor(p, dtype=torch.float32).reshape(1,1, p.shape[0], p.shape[1], p.shape[2])
bx_tensor =torch.tensor(u, dtype=torch.float32).reshape(1,1, bx.shape[0], bx.shape[1], bx.shape[2])
by_tensor =torch.tensor(u, dtype=torch.float32).reshape(1,1, by.shape[0], by.shape[1], by.shape[2])

alpha = 1/dt*2
beta = 1/dx*2
gamma = 1/dx**2

stencil_t = torch.zeros(3,3,3)
stencil = alpha*torch.tensor([[0, -1, 0],
                           [0, 0, 0],
                           [0, 1, 0]], dtype=torch.float32)
                           
stencil_t[:, 1, :] = stencil


stencil_x = torch.zeros(3,3,3)
stencil = beta * torch.tensor([[0, 0, 0],
                           [-1, 0 , 1],
                           [0, 0, 0]], dtype=torch.float32)
                           
stencil_x[1,: , :] = stencil

stencil_y = torch.zeros(3,3,3)
stencil = beta * torch.tensor([[0, 0, 0],
                           [-1, 0 , 1],
                           [0, 0, 0]], dtype=torch.float32)

stencil_y[:, :, 1] = stencil


stencil_xx = torch.zeros(3,3,3)
stencil= gamma * torch.tensor([[0, 0, 0],
                           [1, -2 , 1],
                           [0, 0, 0]], dtype=torch.float32)
                           
stencil_xx[1,: , :] = stencil


stencil_yy = torch.zeros(3,3,3)
stencil = gamma * torch.tensor([[0, 0, 0],
                           [1, -2 , 1],
                           [0, 0, 0]], dtype=torch.float32)
                           
stencil_yy[:, :, 1] = stencil


nine_point_stencil  = torch.tensor([
                           [1, 1, 1],
                           [1, -8, 1],
                           [1, 1, 1]], dtype=torch.float32)


stencil_xx_yy = torch.zeros(3,3,3)
stencil = gamma * nine_point_stencil
stencil_xx_yy[1,: , :] = stencil

stencil_t = stencil_t.view(1, 1, 3, 3, 3)
stencil_x = stencil_x.view(1, 1, 3, 3, 3)
stencil_y =  stencil_y.view(1, 1, 3, 3, 3)
stencil_xx = stencil_xx.view(1, 1, 3, 3, 3)
stencil_yy =  stencil_yy.view(1, 1, 3, 3, 3)
stencil_xx_yy =  stencil_xx_yy.view(1, 1, 3, 3, 3)

# deriv_u = F.conv3d(u_tensor, stencil_t)[0,0] + F.conv3d(u_tensor, stencil_x)[0,0]*u_tensor[...,1:-1, 1:-1, 1:-1] + F.conv3d(u_tensor, stencil_x)[0,0]*v_tensor[...,1:-1, 1:-1, 1:-1]  - nu*(F.conv3d(u_tensor, stencil_xx)[0,0]  + F.conv3d(u_tensor, stencil_yy)[0,0]) + F.conv3d(p_tensor, stencil_x)[0,0]
# deriv_v = F.conv3d(u_tensor, stencil_t)[0,0] + F.conv3d(v_tensor, stencil_x)[0,0]*u_tensor[...,1:-1, 1:-1, 1:-1] + F.conv3d(v_tensor, stencil_x)[0,0]*v_tensor[...,1:-1, 1:-1, 1:-1]  - nu*(F.conv3d(v_tensor, stencil_xx)[0,0]  + F.conv3d(v_tensor, stencil_yy)[0,0]) + F.conv3d(p_tensor, stencil_y)[0,0]

# deriv_u = F.conv3d(u_tensor, stencil_t)[0,0] + F.conv3d(u_tensor, stencil_x)[0,0]*u_tensor[...,1:-1, 1:-1, 1:-1] + F.conv3d(u_tensor, stencil_x)[0,0]*v_tensor[...,1:-1, 1:-1, 1:-1]  - nu*(F.conv3d(u_tensor, stencil_xx_yy)[0,0]) + F.conv3d(p_tensor, stencil_x)[0,0]
# deriv_v = F.conv3d(u_tensor, stencil_t)[0,0] + F.conv3d(v_tensor, stencil_x)[0,0]*u_tensor[...,1:-1, 1:-1, 1:-1] + F.conv3d(v_tensor, stencil_x)[0,0]*v_tensor[...,1:-1, 1:-1, 1:-1]  - nu*(F.conv3d(v_tensor, stencil_xx_yy)[0,0]) + F.conv3d(p_tensor, stencil_y)[0,0]

# deriv_cont = F.conv3d(u_tensor, stencil_x)[0,0] + F.conv3d(v_tensor, stencil_y)[0,0]

# deriv_u = deriv_u[0,0]
# deriv_v = deriv_v[0,0]

div_B = F.conv3d(bx_tensor, stencil_x) + F.conv3d(by_tensor, stencil_y)
# %%
#Test Plots
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm 

fig = plt.figure(figsize=(10, 8))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.5)
idx = 500
ax = fig.add_subplot(3,2,1)
pcm =ax.imshow(u[idx], cmap=cm.coolwarm, extent=[0.0, 1.0, 0.0, 1.0])
ax.title.set_text('Num. Soln. - U')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(3,2,2)
pcm =ax.imshow(deriv_u[idx], cmap=cm.coolwarm,extent=[0.0, 1.0, 0.0, 1.0])
ax.title.set_text('u_residual')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(3,2,3)
pcm =ax.imshow(v[idx], cmap=cm.coolwarm, extent=[0.0, 1.0, 0.0, 1.0])
ax.title.set_text('Num. Soln. - V')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(3,2,4)
pcm =ax.imshow(deriv_v[idx], cmap=cm.coolwarm, extent=[0.0, 1.0, 0.0, 1.0])
ax.title.set_text('v_residual')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(3,2,5)
pcm =ax.imshow(w[idx], cmap=cm.coolwarm, extent=[0.0, 1.0, 0.0, 1.0])
ax.title.set_text('Num. Soln. - W')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(3,2,6)
pcm =ax.imshow(deriv_cont[idx], cmap=cm.coolwarm, extent=[0.0, 1.0, 0.0, 1.0])
ax.title.set_text('w_residual')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

# %%
