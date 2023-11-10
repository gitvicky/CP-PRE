#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

---------------------------------------------------------------------------------------
Testing out the FD gradient estimation of spatio-temporal fields for a 1D Convection Diffusion with varying Diffusion Coeffecients
Demonstrating the Stencil Approach. 

1D Advection Equation 
    u_t = D.u_xx +u.D_x - c.u_x

    c = c #convection velocity 
    D = sin(x/D_damp)) #D_damp - damping factor


Initial Condition: 
    u_0 = np.exp(-(x-mu)**2 / (2*sigma**2)) / np.sqrt(2*np.pi*sigma**2) #Gaussian centered at mu with variance sigma 
---------------------------------------------------------------------------------------
"""

# %% 
#Importing the necessary packages 
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm_gui
import jax.numpy as jnp
import sympy 
import torch 
# %% 
#Obtaining the numerical solution of the 1D Conv-Diff Equation 
from CD_numerical import *

dx = 0.05 #x discretisation
dt = 0.0005 #t discretisation
D_damp = 2*np.pi #Damping Coefficient of the spatial diffusion coefficient
c = 0.5 #Convection Velocity  
mu = 5.0  #Mean of the initial Gaussian
sigma = 0.5 #Variance of the initial Gaussian. 

solver = Conv_Diff_1d(dx, dt, D_damp, c, mu, sigma) 
u_sol, D, D_x = solver.solve()  #solution shape -> t, x


#%%
# %% 
#Obtaining the gradients using np.gradient 
u = u_sol
u_t_np = np.gradient(u_sol, dt, axis=0)
u_x_np = np.gradient(u_sol, dx, axis=1)
u_xx_np = np.gradient(u_x_np, dx, axis=1)

residual_np = u_t_np + D* u_xx_np + u_x_np*D_x - c*u_x_np

#Obtaining the gradients using jax
u_t_jnp = jnp.gradient(u_sol, dt, axis=0)
u_x_jnp = jnp.gradient(u_sol, dx, axis=1)
residual_jnp = u_t_jnp + u_x_jnp
residual_jnp = u_t_np + D* u_xx_np + u_x_jnp*D_x - c*u_x_np


# %%
#Forward Difference.- Automated using sympy 

vars = sympy.symbols('u, x, t')
eqn_str = 'D(u, t, 0) - d*D2(u, x, 1) + c*D(u, x, 1)' # Trying to evaluate the partial derivative of y wrt x

# #D+
# def deriv_FD(u, delta, axis):
#     deriv = np.zeros(u.shape)
#     if axis==0:
#         deriv[:-1, :] = (u[1:, :] - u[:-1, :]) / delta
#         deriv[:, -1] = deriv[:, -2]
#     if axis==1:
#         deriv[:, :-1] = (u[:, 1:] - u[:, :-1]) / delta
#         deriv[-1, :] = deriv[-2, :]
#     return deriv

# fn_FD = sympy.lambdify(vars, eqn_str, {'D': deriv_FD})
# df_FD = fn_FD(u_sol, dt, dx)

#D0
def deriv_FD_CS(u, delta, axis):
    deriv = np.zeros(u.shape)
    if axis==0:
        deriv[:-2, :] = (u[2:, :] - u[:-2, :]) / 2*delta
        deriv[:, -1] = deriv[:, -2]
    if axis==1:
        deriv[:, :-2] = (u[:, 2:] - u[:, :-2]) / 2*delta
        deriv[-1, :] = deriv[-2, :]
    return deriv

def deriv_FD_2(u, delta, axis):
    deriv = np.zeros(u.shape)
    if axis==0:
        deriv[:-2, :] = (u[2:, :] - 2*u[1:-1, :] + u[:-2, :]) / delta**2
        deriv[:, -1] = deriv[:, -2]
    if axis==1:
        deriv[:, :-2] = (u[:, 2:] -2*u[:, 1:-1]+ u[:, :-2]) / delta**2
        deriv[-1, :] = deriv[-2, :]
    return deriv


fn_FD = sympy.lambdify(vars, eqn_str, {'D': deriv_FD_CS, 'D2': deriv_FD_2, 'c':0.5, 'd':1.0})
df_FD = fn_FD(u_sol, dt, dx)


#%%
#Utilising the stencil by way of convolutions using pytorch --- Fixed Diffusion Coefficients 
import torch.nn.functional as F
u_tensor =torch.tensor(u_sol, dtype=torch.float32).reshape(1,1,u_sol.shape[0], u_sol.shape[1])

alpha = (2*dt*D[0]/dx**2)
beta = (c*dt)/dx

stencil = torch.tensor([[0., -1., 0.],
                           [alpha-beta, -2*alpha, alpha+beta],
                           [0., 1., 0.]], dtype=torch.float32)
                           
stencil = stencil.view(1, 1, 3, 3)

deriv_stencil_conv = F.conv2d(u_tensor, stencil)[0,0]

# %%
#Test Plots
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm 

fig = plt.figure(figsize=(10, 8))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.1)

ax = fig.add_subplot(3,2,1)
pcm =ax.imshow(u_sol, cmap=cm.coolwarm, extent=[0.0, 10.0, 2.5, 0])
ax.title.set_text('Num. Soln. ')
ax.set_xlabel('x')
ax.set_ylabel('t')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))


#np.gradient
ax = fig.add_subplot(3,2,3)
pcm =ax.imshow(residual_np[1:-1, 1:-1], cmap=cm.coolwarm, extent=[0.0, 10.0, 2.5, 0])
ax.title.set_text('np gradient')
ax.set_xlabel('x')
ax.set_ylabel('t')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))


#Jax
ax = fig.add_subplot(3,2,4)
pcm =ax.imshow(residual_jnp[1:-1,1:-1], cmap=cm.coolwarm, extent=[0.0, 10.0, 2.5, 0])
ax.title.set_text('jax gradient')
ax.set_xlabel('x')
# ax.set_ylabel('t')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

#Handwritten
ax = fig.add_subplot(3,2,5)
pcm =ax.imshow(df_FD[1:-1,1:-1], cmap=cm.coolwarm, extent=[0.0, 10.0, 2.5, 0])
ax.title.set_text('Handwritten')
ax.set_xlabel('x')
ax.set_ylabel('t')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

#Stencils and Convolutions
ax = fig.add_subplot(3,2,6)
pcm =ax.imshow(deriv_stencil_conv, cmap=cm.coolwarm, extent=[0.0, 10.0, 2.5, 0])
ax.title.set_text('Conv using Stencil')
ax.set_xlabel('x')
# ax.set_ylabel('t')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

# %%
