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
a = 1.0 #u IC params [0.5, 1]
b = 1.0 #v IC params [0.5, 1]
c = 0.5 #p IC params [0.5, 1]
rho, u, v, p, Bx, By, dt = solve(N, boxsize, tEnd, a, b, c)
dx = boxsize/N

gamma = torch.tensor(5/3, dtype=torch.float32)
p_gas = p - 0.5*(Bx**2 + By**2)


#Ignoring the varying dt for the time being --- Needs to be fixed !! 
dt = np.mean(dt) 
#%%
#Utilising the stencil By way of convolutions using pytorch 
import torch.nn.functional as F

rho_tensor =torch.tensor(rho, dtype=torch.float32).reshape(1,1, rho.shape[0], rho.shape[1], rho.shape[2])
u_tensor =torch.tensor(u, dtype=torch.float32).reshape(1,1, u.shape[0], u.shape[1], u.shape[2])
v_tensor =torch.tensor(v, dtype=torch.float32).reshape(1,1, v.shape[0], v.shape[1], v.shape[2])
p_tensor =torch.tensor(p, dtype=torch.float32).reshape(1,1, p.shape[0], p.shape[1], p.shape[2])
Bx_tensor =torch.tensor(Bx, dtype=torch.float32).reshape(1,1, Bx.shape[0], Bx.shape[1], Bx.shape[2])
By_tensor =torch.tensor(By, dtype=torch.float32).reshape(1,1, By.shape[0], By.shape[1], By.shape[2])
p_gas_tensor = torch.tensor(p_gas, dtype=torch.float32).reshape(1,1, p_gas.shape[0], p_gas.shape[1], p_gas.shape[2])

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

deriv_rho = F.conv3d(rho_tensor, stencil_t)[0,0] + v_tensor[...,1:-1, 1:-1, 1:-1] * F.conv3d(rho_tensor, stencil_t)[0,0] + rho_tensor[...,1:-1, 1:-1, 1:-1] * F.conv3d(u_tensor, stencil_x)[0,0] + v_tensor[...,1:-1, 1:-1, 1:-1] * F.conv3d(rho_tensor, stencil_y)[0,0] + rho_tensor[...,1:-1, 1:-1, 1:-1] * F.conv3d(v_tensor, stencil_y)
deriv_u = F.conv3d(u_tensor, stencil_t)[0,0] + u_tensor[...,1:-1, 1:-1, 1:-1] * F.conv3d(u_tensor, stencil_x)[0,0] + (1/rho_tensor[...,1:-1, 1:-1, 1:-1]) * F.conv3d(p_tensor, stencil_x)[0,0] -2*(Bx_tensor/rho_tensor)[...,1:-1, 1:-1, 1:-1] * F.conv3d(Bx_tensor, stencil_x)[0,0] + v_tensor[...,1:-1, 1:-1, 1:-1] * F.conv3d(u_tensor, stencil_y)[0,0] - (By_tensor/rho_tensor)[...,1:-1, 1:-1, 1:-1] * F.conv3d(Bx_tensor, stencil_y)[0,0] - (Bx_tensor/rho_tensor)[...,1:-1, 1:-1, 1:-1] * F.conv3d(By_tensor, stencil_y)[0,0]
deriv_v = F.conv3d(v_tensor, stencil_t)[0,0] + u_tensor[...,1:-1, 1:-1, 1:-1] * F.conv3d(v_tensor, stencil_x)[0,0] + (1/rho_tensor[...,1:-1, 1:-1, 1:-1]) * F.conv3d(p_tensor, stencil_y)[0,0] -2*(By_tensor/rho_tensor)[...,1:-1, 1:-1, 1:-1] * F.conv3d(By_tensor, stencil_y)[0,0] + v_tensor[...,1:-1, 1:-1, 1:-1] * F.conv3d(v_tensor, stencil_y)[0,0] - (By_tensor/rho_tensor)[...,1:-1, 1:-1, 1:-1] * F.conv3d(Bx_tensor, stencil_x)[0,0] - (Bx_tensor/rho_tensor)[...,1:-1, 1:-1, 1:-1] * F.conv3d(By_tensor, stencil_y)[0,0]
deriv_P = F.conv3d(p_tensor, stencil_t)[0,0] + (gamma*p_gas_tensor[...,1:-1, 1:-1, 1:-1] + By_tensor[...,1:-1, 1:-1, 1:-1]**2)  * F.conv3d(u_tensor, stencil_x)[0,0]  - Bx_tensor[...,1:-1, 1:-1, 1:-1] * By_tensor[...,1:-1, 1:-1, 1:-1] * F.conv3d(v_tensor, stencil_x)[0,0] + u_tensor[...,1:-1, 1:-1, 1:-1] * F.conv3d(p_tensor, stencil_x)[0,0] + (gamma -2)*(Bx_tensor[...,1:-1, 1:-1, 1:-1]*u_tensor[...,1:-1, 1:-1, 1:-1] + By_tensor[...,1:-1, 1:-1, 1:-1]*v_tensor[...,1:-1, 1:-1, 1:-1]) * F.conv3d(Bx_tensor, stencil_x)[0,0] -  By_tensor[...,1:-1, 1:-1, 1:-1] * Bx_tensor[...,1:-1, 1:-1, 1:-1] * F.conv3d(u_tensor, stencil_y)[0,0] + (gamma*p_gas_tensor[...,1:-1, 1:-1, 1:-1] + Bx_tensor[...,1:-1, 1:-1, 1:-1]**2)  * F.conv3d(v_tensor, stencil_y)[0,0] + v_tensor[...,1:-1, 1:-1, 1:-1] * F.conv3d(p_tensor, stencil_y)[0,0] + (gamma -2)*(Bx_tensor[...,1:-1, 1:-1, 1:-1]*u_tensor[...,1:-1, 1:-1, 1:-1] + By_tensor[...,1:-1, 1:-1, 1:-1]*v_tensor[...,1:-1, 1:-1, 1:-1]) * F.conv3d(By_tensor, stencil_y)[0,0]
deriv_Bx = F.conv3d(Bx_tensor, stencil_t)[0,0] - By_tensor[...,1:-1, 1:-1, 1:-1] * F.conv3d(u_tensor, stencil_y)[0,0] + Bx_tensor[...,1:-1, 1:-1, 1:-1] * F.conv3d(v_tensor, stencil_y)[0,0] - v_tensor[...,1:-1, 1:-1, 1:-1] * F.conv3d(Bx_tensor, stencil_y)[0,0] + u_tensor[...,1:-1, 1:-1, 1:-1] * F.conv3d(By_tensor, stencil_y)[0,0]
deriv_By = F.conv3d(By_tensor, stencil_t)[0,0] + By_tensor[...,1:-1, 1:-1, 1:-1] * F.conv3d(u_tensor, stencil_x)[0,0] - Bx_tensor[...,1:-1, 1:-1, 1:-1] * F.conv3d(v_tensor, stencil_x)[0,0] - v_tensor[...,1:-1, 1:-1, 1:-1] * F.conv3d(Bx_tensor, stencil_x)[0,0] + u_tensor[...,1:-1, 1:-1, 1:-1] * F.conv3d(By_tensor, stencil_x)[0,0]

div_B = F.conv3d(Bx_tensor, stencil_x) + F.conv3d(By_tensor, stencil_y)
# %%
#Test Plots
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm 

# field = p
# deriv = deriv_P[0,0]

# fig = plt.figure(figsize=(10, 8))
# plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.5)
# idx = -1
# ax = fig.add_subplot(3,2,1)
# pcm =ax.imshow(field[idx], cmap=cm.coolwarm, extent=[0.0, 1.0, 0.0, 1.0])
# ax.title.set_text('Num. Soln.')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.1)
# cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

# ax = fig.add_subplot(3,2,2)
# pcm =ax.imshow(deriv[idx], cmap=cm.coolwarm,extent=[0.0, 1.0, 0.0, 1.0])
# ax.title.set_text('Residual')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.1)
# cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

# ax = fig.add_subplot(3,2,3)
# pcm =ax.imshow(v[idx], cmap=cm.coolwarm, extent=[0.0, 1.0, 0.0, 1.0])
# ax.title.set_text('Num. Soln. - V')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.1)
# cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

# ax = fig.add_subplot(3,2,4)
# pcm =ax.imshow(deriv_v[idx], cmap=cm.coolwarm, extent=[0.0, 1.0, 0.0, 1.0])
# ax.title.set_text('v_residual')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.1)
# cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

# ax = fig.add_subplot(3,2,5)
# pcm =ax.imshow(w[idx], cmap=cm.coolwarm, extent=[0.0, 1.0, 0.0, 1.0])
# ax.title.set_text('Num. Soln. - W')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.1)
# cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

# ax = fig.add_subplot(3,2,6)
# pcm =ax.imshow(deriv_cont[idx], cmap=cm.coolwarm, extent=[0.0, 1.0, 0.0, 1.0])
# ax.title.set_text('w_residual')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.1)
# cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

# # %%

# %%
# %% 
import matplotlib as mpl

fig = plt.figure()
mpl.rcParams['figure.figsize']=(12, 30)
plt.figure()
num_vars = 6
vars = ['rho', 'u', 'v', 'P', 'Bx', 'By']
fields = [rho[...,1:-1, 1:-1, 1:-1], u[...,1:-1, 1:-1, 1:-1], v[...,1:-1, 1:-1, 1:-1], p[...,1:-1, 1:-1, 1:-1], Bx[...,1:-1, 1:-1, 1:-1], By[...,1:-1, 1:-1, 1:-1]]
derivs = [deriv_rho, deriv_u, deriv_v, deriv_P, deriv_Bx, deriv_By]
idx = -1 
for ii, field in enumerate(fields):
    jj = ii+ii+1
    
    ax = fig.add_subplot(6,2,jj)
    pcm =ax.imshow(field[idx], cmap=cm.coolwarm, extent=[0.0, 1.0, 0.0, 1.0])
    ax.title.set_text("Num. Soln. - " + vars[ii])
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))

    ax = fig.add_subplot(6,2,jj+1)
    pcm =ax.imshow(derivs[ii][0,0][idx], cmap=cm.coolwarm,extent=[0.0, 1.0, 0.0, 1.0])
    ax.title.set_text('Residual - ' + vars[ii])
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
# %%
