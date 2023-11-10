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

solver= Navier_Stokes_2d(400, 0.0, 1.0, 0.001, 0.001, 1) # N, t, tEnd, dt, nu, L
u, v, p, w, x, t = solver.solve()

# %% 
dt = 0.001

# %%
# #Forward Difference.- Automated using sympy 

vars = sympy.symbols('u, t, x, y')
eqn_str = 'D2(u, t, 0) - D2(u, x, 1) - D2(u, y, 2)' 

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
u_tensor =torch.tensor(u, dtype=torch.float32).reshape(1,1,u_sol.shape[0], u.shape[1], u.shape[2])

alpha = 1/dt**2
beta = 1/dx**2

stencil_time = torch.zeros(3,3,3)
stencil_t = alpha*torch.tensor([[0, 1, 0],
                           [0, -2, 0],
                           [0, 1, 0]], dtype=torch.float32)
                           
stencil_time[:, 1, :] = stencil_t


stencil_xx = torch.zeros(3,3,3)
stencil_x = beta * torch.tensor([[0, 0, 0],
                           [1, -2 , 1],
                           [0, 0, 0]], dtype=torch.float32)
                           
stencil_xx[1,: , :] = stencil_x


stencil_yy = torch.zeros(3,3,3)
stencil_y = beta * torch.tensor([[0, 0, 0],
                           [1, -2 , 1],
                           [0, 0, 0]], dtype=torch.float32)
                           
stencil_yy[:, :, 1] = stencil_y


stencil_time = stencil_time.view(1, 1, 3, 3, 3)
stencil_xx = stencil_xx.view(1, 1, 3, 3, 3)
stencil_yy =  stencil_yy.view(1, 1, 3, 3, 3)

deriv_stencil = F.conv3d(u_tensor, stencil_time)[0,0] - F.conv3d(u_tensor, stencil_xx)[0,0] - F.conv3d(u_tensor, stencil_yy)[0,0]

# %%
#Test Plots
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm 

fig = plt.figure(figsize=(10, 8))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.1)
idx = 100
ax = fig.add_subplot(3,2,1)
pcm =ax.imshow(u_sol[idx], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0])
ax.title.set_text('Numerical Soln. ')
ax.set_xlabel('x')
ax.set_ylabel('t')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

# ax = fig.add_subplot(3,2,2)
# pcm =ax.imshow(u_sol, cmap=cm.coolwarm,extent=[-1.0, 1.0, -1.0, 1.0])
# ax.title.set_text('Numerical')
# ax.set_xlabel('x')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.1)
# cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

#Handwritten
ax = fig.add_subplot(3,2,3)
pcm =ax.imshow(df_FD[idx][1:-1,1:-1], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0])
ax.title.set_text('Handwritten')
ax.set_xlabel('x')
ax.set_ylabel('t')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

#Stencils and Convolutions
ax = fig.add_subplot(3,2,4)
pcm =ax.imshow(deriv_stencil[idx], cmap=cm.coolwarm, extent=[-1.0, 1.0, -1.0, 1.0])
ax.title.set_text('Conv using Stencil')
ax.set_xlabel('x')
# ax.set_ylabel('t')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

# %% 
#Visualising the 3D Tensor

#Example structure: 
# plt.rcParams["figure.figsize"] = [7.00, 3.50]
# plt.rcParams["figure.autolayout"] = True
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# data = np.random.random(size=(3, 3, 3))
# z, x, y = data.nonzero()
# ax.scatter(x, y, z, c=z, alpha=1)
# plt.show()


# %%
