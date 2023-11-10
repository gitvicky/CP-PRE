#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

---------------------------------------------------------------------------------------
Testing out the FD gradient estimation of spatio-temporal fields for a 1D Advection Equation
Comparing Jax and Numpy implementations. 

1D Advection Equation 
    U_t + v U_x = 0
where, v = 1.0

Exact Solver : 
  U(x,t) = exp(-200*(x-xc-v*t).^2)

  xc - location of the center -- 0.25

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
from Advection_1D import *
#Obtaining the Analytical and FD solution of the 1D Advection Equation. 
x_discretisation, t_end = 200, 1.0
solver = Advection_1d(x_discretisation, t_end)
x, t, u_sol, u_exact = solver.solve()  #solution shape -> t, c

dt = solver.dt
dx = solver.dx
v = 1.0 #Convection Velocity

#Truncated solutions if needed. 
# u_sol = u_sol[:50] 
# u_exact = u_exact[:50]
# %%
#Exact solution of tthe generic form u(x,t) = u0*(x − wt)
exact_fn = lambda it : np.exp(-200*(x-solver.xc - v*it*dt)**2)
du_dx =  lambda it : - exact_fn(it) * (-400*it*dt + 400*x - 100)
du_dt = lambda it :  exact_fn(it) * (-400*it*dt + 400*x - 100)
exact_residual = lambda it : du_dt(it) + du_dx(it) 

#%%
# %% 
#Obtaining the gradients using np.gradient 
u_val = u_sol# Either on the numerical soln or on the analytical soln. 
u_t_np = np.gradient(u_val, dt, axis=0)
u_x_np = np.gradient(u_val, dx, axis=1)
residual_np = u_t_np + u_x_np

#Obtaining the gradients using jax
u_t_jnp = jnp.gradient(u_val, dt, axis=0)
u_x_jnp = jnp.gradient(u_val, dx, axis=1)
residual_jnp = u_t_jnp + u_x_jnp


#Checking the accuracy of the gradients 
plt.figure()
idx = 10 
plt.plot(du_dt(idx), label='analytical')
plt.plot(u_t_jnp[idx-1], label='jax')
plt.plot(u_t_np[idx-1], label='numpy')
plt.title('u_t')
plt.legend()

plt.figure()
idx = 10 
plt.plot(du_dx(idx), label='analytical')
plt.plot(u_x_jnp[idx-1], label='jax')
plt.plot(u_x_np[idx-1], label='numpy')
plt.title('u_x')
plt.legend()
# %%
#Forward Difference.- Automated using sympy 

vars = sympy.symbols('u, x, t')
eqn_str = 'D(u, t, 0) + 1.0*D(u, x, 1)' # Trying to evaluate the partial derivative of y wrt x

#D+
def deriv_FD(u, delta, axis):
    deriv = np.zeros(u.shape)
    if axis==0:
        deriv[:-1, :] = (u[1:, :] - u[:-1, :]) / delta
        deriv[:, -1] = deriv[:, -2]
    if axis==1:
        deriv[:, :-1] = (u[:, 1:] - u[:, :-1]) / delta
        deriv[-1, :] = deriv[-2, :]
    return deriv

fn_FD = sympy.lambdify(vars, eqn_str, {'D': deriv_FD})
df_FD = fn_FD(u_val, dt, dx)

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

fn_FD = sympy.lambdify(vars, eqn_str, {'D': deriv_FD})
df_FD = fn_FD(u_val, dt, dx)


fn_FD = sympy.lambdify(vars, eqn_str, {'D': deriv_FD_CS})
df_FD_CS = fn_FD(u_val, dt, dx)


#%%
#Utilising the stencil by way of convolutions using pytorch 
import torch.nn.functional as F
u_tensor =torch.tensor(u_val, dtype=torch.float32).reshape(1,1,u_val.shape[0], u_val.shape[1])
stencil_CT_CS = torch.tensor([[0., -1./(2*dt), 0.],
                           [-1./(2*dx), 0., 1./(2*dx)],
                           [0., 1./(2*dt), 0.]])
                           
stencil_CT_CS = stencil_CT_CS.view(1, 1, 3, 3)


stencil_FT_CS = torch.tensor([[0., -1./dt, 0.],
                           [-1./(2*dx), 1./dt, 1./(2*dx)],
                           [0., 0., 0.]])
                           
stencil_FT_CS = stencil_FT_CS.view(1, 1, 3, 3)


deriv_stencil_conv = F.conv2d(u_tensor, stencil_CT_CS)[0,0]

# %%
#Test Plots
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm 

fig = plt.figure(figsize=(10, 8))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.1)

ax = fig.add_subplot(3,2,1)
pcm =ax.imshow(u_exact, cmap=cm.coolwarm, extent=[0.0, 2.0, 1.0, 0])
ax.title.set_text('Analytical')
ax.set_xlabel('x')
ax.set_ylabel('t')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(3,2,2)
pcm =ax.imshow(u_sol, cmap=cm.coolwarm, extent=[0.0, 2.0, 1.0, 0])
ax.title.set_text('Numerical')
ax.set_xlabel('x')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))


#np.gradient
ax = fig.add_subplot(3,2,3)
pcm =ax.imshow(np.abs(u_sol - u_exact), cmap=cm.coolwarm, extent=[0.0, 2.0, 1.0, 0])
ax.title.set_text('Abs. Error')
ax.set_xlabel('x')
ax.set_ylabel('t')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))


#MAE
ax = fig.add_subplot(3,2,4)
pcm =ax.imshow(residual_jnp[1:-1,1:-1], cmap=cm.coolwarm, extent=[0.0, 2.0, 1.0, 0])
ax.title.set_text('jax gradient')
ax.set_xlabel('x')
# ax.set_ylabel('t')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

#Handwritten
ax = fig.add_subplot(3,2,5)
pcm =ax.imshow(df_FD_CS[1:-1,1:-1], cmap=cm.coolwarm, extent=[0.0, 2.0, 1.0, 0])
ax.title.set_text('Handwritten')
ax.set_xlabel('x')
ax.set_ylabel('t')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

#Stencils and Convolutions
ax = fig.add_subplot(3,2,6)
pcm =ax.imshow(deriv_stencil_conv, cmap=cm.coolwarm, extent=[0.0, 2.0, 1.0, 0])
ax.title.set_text('Conv using Stencil')
ax.set_xlabel('x')
# ax.set_ylabel('t')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))
# %%
#If the stencil was invertible, trying out the transposed convolution

#Slightly modified stencil to make it invertible -- Seeing if doing a transposed convolution could on the inverse of the convolution kernel will help retrieve the field. 
stencil = torch.tensor([[1., -1./(2), 0.],
                           [-1./(2), 0., 1./(2*dx)],
                           [1., 1./(2), 0.]])
stencil = stencil.view(1, 1, 3, 3)

print(torch.linalg.norm(stencil))

deriv_stencil_conv = F.conv2d(u_tensor, stencil)

inv_stencil = torch.linalg.inv(stencil)

u_trans_conv= F.conv_transpose2d(deriv_stencil_conv, inv_stencil.view(1,1,3,3))[0,0]
# %%
