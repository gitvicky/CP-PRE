#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

---------------------------------------------------------------------------------------
Testing out the FD gradient estimation of spatio-temporal fields for a 2D Navier-Stokes Incompressible Flow
Demonstrating the Stencil Approach. 

NS Equation - Velocity formulation 
------------------------------------------------------
v_t + (v.nabla) v = nu * nabla^2 v + nabla P
div(v) = 0

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

from NS_Numerical import *

solver= Navier_Stokes_2d(400, 0.0, 1.0, 0.001, 0.001, 1, 1.0, 1.0) # N, t, tEnd, dt, nu, L, a , b  #(a and b are multiplier for the IC - ranging from 0.1 to 1, 1 being used by Philip)
u, v, p, w, x, t, err = solver.solve()

# %% 
nu = 0.001
dt = 0.001
dx = x[-1] - x[-2]
# %%
# #Forward Difference.- Automated using sympy 

# vars = sympy.symbols('u, t, x, y')
# eqn_str = 'D2(u, t, 0) - D2(u, x, 1) - D2(u, y, 2)' 

# #D0
# def deriv_FD(u, delta, axis):
#     deriv = np.zeros(u.shape)
#     if axis==0:
#         deriv[:-2, :, :] = (u[2:, :, :] - u[:-2, :, :]) / 2*delta
#     if axis==1:
#         deriv[:, :-2, :] = (u[:, 2:, :] - u[:, :-2, :]) / 2*delta
#     if axis==2:
#         deriv[:, :, :-2] = (u[:, :, 2:] - u[:, :, :-2]) / 2*delta
#     return deriv

# def deriv_FD_2(u, delta, axis):
#     deriv = np.zeros(u.shape)
#     if axis==0:
#         deriv[:-2, :, :] = (u[2:, :, :] - 2*u[1:-1, :, :] + u[:-2, :, :]) / delta**2
#     if axis==1:
#         deriv[:, :-2, :] = (u[:, 2:, :] -2*u[:, 1:-1, :]+ u[:, :-2, :]) / delta**2
#     if axis==2:
#         deriv[:, :, :-2] = (u[:, :, 2:] -2*u[:, :, 1:-1] +  u[:, :, :-2]) / delta**2
#     return deriv


# fn_FD = sympy.lambdify(vars, eqn_str, {'D': deriv_FD, 'D2': deriv_FD_2})
# df_FD = fn_FD(u_sol, dt, dx, dy)


#%%
#Utilising the stencil by way of convolutions using pytorch 
import torch.nn.functional as F
u_tensor =torch.tensor(u, dtype=torch.float32).reshape(1,1, u.shape[0], u.shape[1], u.shape[2])
v_tensor =torch.tensor(v, dtype=torch.float32).reshape(1,1, v.shape[0], v.shape[1], v.shape[2])
p_tensor =torch.tensor(p, dtype=torch.float32).reshape(1,1, p.shape[0], p.shape[1], p.shape[2])

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

# #Combining the x and y gradients together. 
# stencil_xy = torch.zeros(3,3,3)
# stencil_xy[1, :, :] = stencil
# stencil_xy[:, :, 1] = stencil

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

deriv_u = F.conv3d(u_tensor, stencil_t)[0,0] + F.conv3d(u_tensor, stencil_x)[0,0]*u_tensor[...,1:-1, 1:-1, 1:-1] + F.conv3d(u_tensor, stencil_y)[0,0]*v_tensor[...,1:-1, 1:-1, 1:-1]  - nu*(F.conv3d(u_tensor, stencil_xx_yy)[0,0]) + F.conv3d(p_tensor, stencil_x)[0,0]
deriv_v = F.conv3d(v_tensor, stencil_t)[0,0] + F.conv3d(v_tensor, stencil_x)[0,0]*u_tensor[...,1:-1, 1:-1, 1:-1] + F.conv3d(v_tensor, stencil_y)[0,0]*v_tensor[...,1:-1, 1:-1, 1:-1]  - nu*(F.conv3d(v_tensor, stencil_xx_yy)[0,0]) + F.conv3d(p_tensor, stencil_y)[0,0]

deriv_cont = F.conv3d(u_tensor, stencil_x)[0,0] + F.conv3d(v_tensor, stencil_y)[0,0]

deriv_u = deriv_u[0,0]
deriv_v = deriv_v[0,0]

# deriv_stencil =  deriv_cont
# deriv_stencil = deriv_stencil[0,0]
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
