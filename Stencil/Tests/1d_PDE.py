#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

---------------------------------------------------------------------------------------
Testing Lambdify features of Sympy for evaluating FD across spatio-temporal grids.  
                                    ------------------      
Testing out a structure for a 1D PDE. 
---------------------------------------------------------------------------------------


"""
# %% 
#Importing the necessary packages 
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import torch
import torch.nn as nn
import sympy 

# %% 
'''
# 1D Advection Equation 
    U_t + v U_x = 0
where, v = 1.0

Exact Solver : 
  U(x,t) = exp(-200*(x-xc-v*t).^2)

  xc - location of the center -- 0.25

'''
from Advection_1D import *
#Obtaining the exact and FD solution of the 1D Advection Equation. 
x_discretisation, t_end = 200, 1.0
solver = Advection_1d(x_discretisation, t_end)
x, t, u_sol, u_exact = solver.solve()

dt = solver.dt
dx = solver.dx
v = 1.0 #Convection Velocity

# u_sol = u_sol[:50]
# u_exact = u_exact[:50]
# %%

#Exact solution of tthe generic form u(x,t) = u0*(x − wt)
exact_fn = lambda it : np.exp(-200*(x-solver.xc - v*it*dt)**2)
du_dx =  lambda it : - exact_fn(it) * (-400*it*dt + 400*x - 100)
du_dt = lambda it :  exact_fn(it) * (-400*it*dt + 400*x - 100)
exact_deriv = lambda it : du_dt(it) + du_dx(it) 
# %% 
#Setting the Differential Functions
vars = sympy.symbols('u, v, x, t') #u format needs to be batch size, t, x
eqn_str = 'D_t(u, t) + v*D_x(u, x)'

def FD_t(dep, ind):
    deriv = torch.zeros(dep.shape)
    deriv[:-1, :] = (dep[1:, :] - dep[:-1, :]) / dt
    deriv[:, -1] = deriv[:, -2]
    return deriv

def FD_x(dep, ind):
    deriv = torch.zeros(dep.shape)
    deriv[:, :-1] = (dep[:, 1:] - dep[:, :-1]) / dx
    deriv[-1, :] = deriv[-2, :]
    return deriv


fn_FD = sympy.lambdify(vars, eqn_str, {'D_t': FD_t, 'D_x': FD_x})
df_FD = fn_FD(torch.tensor(u_sol), torch.tensor(v), torch.tensor(x), torch.tensor(t))

# %%
#Comparing obtained derivatives : 

#Using np gradient 
grads = np.gradient(u_sol, dt, dx)

idx = 12
plt.figure()
plt.plot(du_dx(idx), label='Exact')
plt.plot(grads[1][idx, :], label='np')
plt.plot(FD_x(torch.tensor(u_sol), x)[idx, :], label='Calc')
plt.legend()
plt.title('u_x')

plt.figure()
plt.plot(du_dt(idx), label='Exact')
plt.plot(grads[0][idx, :], label='np')
plt.plot(FD_t(torch.tensor(u_sol), t)[idx, :], label='Calc')
plt.legend()
plt.title('u_t')

# %%
plt.figure()
plt.plot(torch.mean(df_FD[2:], axis=1), label='Residual Error')
plt.plot(np.mean(u_sol[2:]- u_exact[2:], axis=1), label='Solution Error')
plt.xlabel('Time')
plt.ylabel('Error')
plt.legend()
# %%
# %%
#Arbitraily testing 
x = np.linspace(0, 100, 1000)
u = np.sin(x)
u_x = np.cos(x)
u_xx = -np.sin(x)
# %%
u_x_grads = np.gradient(u, x)
u_xx_grads = np.gradient(u_x_grads, x)

# %%
plt.figure()
plt.plot(u_x, label='u_x - Actual')
plt.plot(u_xx, label='u_xx - Actual')
plt.plot(u_x_grads, label='u_x - Calc')
plt.plot(u_xx_grads, label='u_xx - Calc')
plt.legend()
# %%
