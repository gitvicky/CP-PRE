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

def deriv_AD(dep, ind):
     return torch.autograd.grad(dep,ind, grad_outputs=torch.ones_like(dep), create_graph=True, allow_unused=True)[0]

fn_FD = sympy.lambdify(vars, eqn_str, {'D_t': FD_t, 'D_x': FD_x})
df_FD = fn_FD(torch.tensor(u_sol), torch.tensor(v), torch.tensor(x), torch.tensor(t))

# %%
#Comparing obtained derivatives : 

idx = 12
plt.figure()
plt.plot(du_dx(idx), label='Exact')
plt.plot(FD_x(torch.tensor(u_sol), x)[idx, :], label='Calc')
plt.legend()
plt.title('u_x')

plt.figure()
plt.plot(du_dt(idx), label='Exact')
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
