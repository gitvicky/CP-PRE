#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
26th March, 2024

Tests to determine the position and axis of the stencils with respect to the spatio-temporal grids
Initially deploying it on a random trig function adn then. 
Deploying the Scheme on the 2D heat equation with an analytical solution. 
"""

# %% 
from sympy import Symbol, sin, cos, diff, lambdify
from sympy import *
import numpy as np
from tqdm import tqdm 

def trig_func(x_mesh, y_mesh, t_val):
    # Define the symbols
    x = Symbol('x')
    y = Symbol('y')
    t = Symbol('t')
    
    # Define the function
    # f = sin(x**2/t) + cos(2*y/t)#trig
    f =( 1 / (4 * pi * 0.1 * t)) * exp(-(x**2 + y**2) / (4 * 0.1 * t))#heat equation
    
    # Create lambdified functions for the function and derivatives
    f_lambda = lambdify((x, y, t), f, 'numpy')
    df_dx_lambda = lambdify((x, y, t), diff(f, x, 2), 'numpy')
    df_dy_lambda = lambdify((x, y, t), diff(f, y, 2), 'numpy')
    df_dt_lambda = lambdify((x, y, t), diff(f, t), 'numpy')
    
    # Evaluate the function and derivatives using the meshgrids and t value
    f_val = f_lambda(x_mesh, y_mesh, t_val)
    df_dxx_val = df_dx_lambda(x_mesh, y_mesh, t_val)
    df_dyy_val = df_dy_lambda(x_mesh, y_mesh, t_val)
    df_dt_val = df_dt_lambda(x_mesh, y_mesh, t_val)
    
    return f_val, df_dxx_val, df_dyy_val, df_dt_val

# Create a grid of points
grid_res = 512
x = np.linspace(-1, 1, grid_res)
y = np.linspace(-1, 1, grid_res)
X, Y = np.meshgrid(x, y)

dx = x[-1] - x[-2]
dt = 0.001
t_range = np.arange(0.1,0.2+dt,dt)
t_res = len(t_range)


u = []
u_t = []
u_xx = []
u_yy = []

for i, t in tqdm(enumerate(t_range)):
    f_val, df_dxx_val, df_dyy_val, df_dt_val = trig_func(X, Y, t)
    u.append(f_val)
    u_xx.append(df_dxx_val)
    u_yy.append(df_dyy_val)
    u_t.append(df_dt_val)

u = np.asarray(u)
u_t = np.asarray(u_t)
u_xx = np.asarray(u_xx)
u_yy = np.asarray(u_yy)

# %% 
#Taking the derivatives using the Finite Difference Scheme with the Convolutional Kernels. 
import torch 
from torch.nn import functional as F
#Some tests to ensure that the kernels are aligned to the correct axis. 
#Compaing the Laplacian Operator implemented in 2D and 3D 

alpha = 1/(dx**2)
# alpha = 1 
beta = 1/(2*dt)

three_p_stencil = torch.tensor([[0, 1, 0],
                           [0, -2, 0],
                           [0, 1, 0]], dtype=torch.float32)

laplacian_stencil_2nd = torch.tensor([[0., -1., 0.],
                       [-1., 4., -1.],
                       [0., -1., 0.]])

laplacian_stencil_4th = torch.tensor([[0, 0, -1/12, 0, 0],
                                       [0, 0, 4/3, 0, 0],
                                       [-1/12, 4/3, -5/2, 4/3, -1/12],
                                       [0, 0, 4/3, 0, 0],
                                       [0, 0, -1/12, 0, 0]]
                                       )

laplacian_stencil_6th = torch.tensor([[0, 0, 0, 1/90, 0, 0, 0],
                                      [0, 0, 0, -3/20, 0, 0, 0],
                                      [0, 0, 0, 3/2, 0, 0, 0],
                                      [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90],
                                      [0, 0, 0, 3/2, 0, 0, 0],
                                      [0, 0, 0, -3/20, 0, 0, 0],
                                      [0, 0, 0, 1/90, 0, 0, 0]], dtype=torch.float32)

laplacian_stencil_8th = torch.tensor([
    [-9,    0,     0,     0,     0,     0,     0,     0,    -9],
    [0,   128,     0,     0,     0,     0,     0,   128,     0],
    [0,     0, -1008,     0,     0,     0, -1008,     0,     0],
    [0,     0,     0,  8064,     0,  8064,     0,     0,     0],
    [0,     0,     0,     0, -14350,     0,     0,     0,     0],
    [0,     0,     0,  8064,     0,  8064,     0,     0,     0],
    [0,     0, -1008,     0,     0,     0, -1008,     0,     0],
    [0,   128,     0,     0,     0,     0,     0,   128,     0],
    [-9,    0,     0,     0,     0,     0,     0,     0,    -9]
    ],dtype=torch.float32)/5040

laplacian_stencil_biharmonic = torch.tensor([[0,0,1,0,0],
                                           [0,2,-8,2,0],
                                           [1,-8,1,-8,1],
                                           [0,2,-8,2,0],
                                           [0,0,1,0,0]], dtype=torch.float32)

CS_stencil_2nd = torch.tensor([[0, -1, 0],
                           [0, 0, 0],
                           [0, 1, 0]], dtype=torch.float32)
                           

u_tensor = torch.tensor(u, dtype=torch.float32)


def conv_deriv_2d(f, stencil):
    return F.conv2d(f.unsqueeze(0).unsqueeze(0), stencil.unsqueeze(0).unsqueeze(0), padding=stencil.shape[0]//2).squeeze()

# u_xx_conv_2d = conv_deriv_2d(u_tensor[-10], three_p_stencil)
# u_yy_conv_2d = conv_deriv_2d(u_tensor[-10], three_p_stencil.T)
# u_xx_yy_conv_2d  = conv_deriv_2d(u_tensor[-10], laplacian_stencil_4th)
# %%
def kernel_3d(stencil, axis):
    kernel_size = stencil.shape[0]
    kernel = torch.zeros(kernel_size, kernel_size, kernel_size)
    if axis == 0:
        kernel[1,:,:] = stencil
    elif axis ==1:
            kernel[:,1,:] = stencil
    elif axis ==2:
            kernel[:,:,1] = stencil
    return kernel

# %%%
def conv_deriv_3d(f, stencil):
    return F.conv3d(f.unsqueeze(0).unsqueeze(0), stencil.unsqueeze(0).unsqueeze(0), padding=(stencil.shape[0]//2, stencil.shape[1]//2, stencil.shape[2]//2)).squeeze()

# u_xx_yy_conv_3d = conv_deriv_3d(u_tensor, kernel_3d(laplacian_stencil_2nd, axis=1))
u_xx_yy_conv_3d = conv_deriv_3d(u_tensor, kernel_3d(laplacian_stencil_4th, axis=0))
# u_xx_yy_conv_3d = conv_deriv_3d(u_tensor, kernel_3d(laplacian_stencil_6th, axis=1))
# u_xx_yy_conv_3d = conv_deriv_3d(u_tensor, kernel_3d(laplacian_stencil_biharmonic, axis=1))
# u_xx_yy_conv_3d = conv_deriv_3d(u_tensor, kernel_3d(alpha*laplacian_stencil_8th, axis=1))
u_t_conv_3d = conv_deriv_3d(u_tensor, kernel_3d(beta*CS_stencil_2nd, axis=2))

# stencil_xx_yy = torch.zeros(3,3,3)
# stencil_xx_yy[: ,1, :] = five_p_stencil
# u_xx_yy_conv_3d = F.conv3d(u_tensor.view(1,1,t_res,grid_res,grid_res), stencil_xx_yy.view(1,1,3,3,3),padding=1)

# stencil_xx_yy = torch.zeros(5,5,5)
# stencil_xx_yy[: ,1, :] = nine_p_stencil
# u_xx_yy_conv_3d = F.conv3d(u_tensor.view(1,1,t_res,grid_res,grid_res), stencil_xx_yy.view(1,1,5,5,5),padding=2)

# stencil_xx_yy = torch.zeros(7,7,7)
# stencil_xx_yy[: ,1, :] = laplacian_stencil_6th
# u_xx_yy_conv_3d = F.conv3d(u_tensor.view(1,1,t_res,grid_res,grid_res), stencil_xx_yy.view(1,1,7,7,7),padding=3)

# stencil_t = torch.zeros(3,3,3)
# stencil_t[: ,:, 1] = two_p_stencil
# u_t_conv_3d = F.conv3d(u_tensor.view(1,1,t_res,grid_res,grid_res), stencil_t.view(1,1,3,3,3),padding=1)

# %%
#Plotting the input, laplace, inverse 
from matplotlib import pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
fig = plt.figure(figsize=(12, 8))
idx = -10

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
pcm =ax.imshow(u_xx_yy_conv_3d[idx][1:-1, 1:-1], cmap=cm.coolwarm)#, vmin=mini, vmax=maxi)
ax.title.set_text(r'$(u_{xx} + u_{yy})$')
ax.set_xlabel('x')
ax.set_ylabel('CK as FDS')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(2,2,2)
pcm =ax.imshow(u_t_conv_3d[idx], cmap=cm.coolwarm)#,  vmin=mini, vmax=maxi)
ax.title.set_text(r'$u_t$')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(2,2,3)
pcm =ax.imshow((u_xx[idx] + u_yy[idx]), cmap=cm.coolwarm)#, vmin=mini, vmax=maxi)
ax.title.set_text(r'$(u_{xx} + u_{yy})$')
ax.set_xlabel('x')
ax.set_ylabel('Analytical')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(2,2,4)
pcm =ax.imshow(u_t[idx], cmap=cm.coolwarm)#,  vmin=mini, vmax=maxi)
ax.title.set_text(r'$u_t$')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

# %%
