#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

---------------------------------------------------------------------------------------
Testing Lambdify features of Sympy for evaluating FD across spatio-temporal grids.  
---------------------------------------------------------------------------------------

Defining the variables of interest --> Taking in the PDE residual as string --> Converting that into a function to be evaluated using sympy --> evaluating the residual on the newly created pseudo-functions with tensoral field values. 

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
#Testing Sympy's functionalities. 
#Bare function evaluation 
from sympy.abc import x #Defining the variables
from sympy import lambdify


fn = lambdify(x, 'x+1')

print(fn(2))
# %%
#Testing out gradient evaluation - converting string to a function as used for tf-pde - identifying autograd derivatives
#https://github.com/gitvicky/tf-pde/

#Here, we define the variables of interest, define the relationship between them as a str and then evaluate them for various data values .
from sympy.parsing.sympy_parser import parse_expr

vars = sympy.symbols('y, x')
eqn_str = 'D(y, x)' # Trying to evaluate the partial derivative of y wrt x

def deriv(dep, ind):
     return torch.autograd.grad(dep,ind, grad_outputs=torch.ones_like(dep), create_graph=True, allow_unused=True)[0]

fn = sympy.lambdify(vars, eqn_str, {'D': deriv})

# %%
a = torch.tensor(3.0, requires_grad=True)
b = a**2
fn(b, a)

a = torch.arange(0,10, requires_grad=True, dtype=float)
b = a**2
fn(b, a)

print('Actual Derivative : ' + str(2*a))
print('Calculated Derivative (AD : ' + str(fn(b, a)))
# %%
#Now lets try and implement it for a finite difference derivative

vars = sympy.symbols('y, x')
eqn_str = 'D(y, x)' # Trying to evaluate the partial derivative of y wrt x

def deriv(dep, ind):
     dx = (ind[-1] - ind[0]) / (len(ind) -1 )
     return (dep[1:] - dep[:-1]) / dx

def center_deriv(dep, ind):
    dx = (ind[-1] - ind[0]) / (len(ind) -1 )
    return (dep[2:] -2*dep[1:-1] + dep[:-2]) / dx**2 #Some issue is here for this
     
fn = sympy.lambdify(vars, eqn_str, {'D': deriv})
fn_CS = sympy.lambdify(vars, eqn_str, {'D': center_deriv})

# %% 

a = torch.arange(0,10, requires_grad=True, dtype=float)
b = a**2

print('Actual Derivative : ' + str(2*a))
print('Calculated Derivative (FD) : ' + str(fn(b,a)))
print('Calculated Derivative (FD CS) : ' + str(fn_CS(b,a)))

# %%
#Spectral Derviatives --- Needs to look into this as the derivatives are rather off. 

vars = sympy.symbols('y, x')
eqn_str = 'D(y, x)' # Trying to evaluate the partial derivative of y wrt x

def deriv(dep, ind):
    dx = (ind[-1] - ind[0]) / (len(ind) -1 )
    kappa = torch.fft.fftfreq(len(dep), d=float(dx))
    hat = torch.fft.fft(dep)
    d_hat = (1j)*kappa*hat
    return torch.fft.ifft(d_hat).real

fn = sympy.lambdify(vars, eqn_str, {'D': deriv})

# %%

a = torch.arange(0,10, requires_grad=True, dtype=float)
b = a**2

print('Actual Derivative : ' + str(2*a))
print('Calculated Derivative (Spectral) : ' + str(fn(b,a)))
# %%
#Spectral but using Numpy

vars = sympy.symbols('y, x')
eqn_str = 'D(y, x)' # Trying to evaluate the partial derivative of y wrt x

def deriv(dep, ind):
    dx = (ind[-1] - ind[0]) / (len(ind) -1)
    kappa = np.fft.fftfreq(len(dep), d=float(dx))
    hat = np.fft.fft(dep)
    d_hat = (1j)*kappa*hat
    return np.fft.ifft(d_hat).real

fn = sympy.lambdify(vars, eqn_str, {'D': deriv})

# %%

a = np.arange(0,10)
b = a**2

print('Actual Derivative : ' + str(2*a))
print('Calculated Derivative (Spectral) : ' + str(fn(b,a)))
# %%
#Comparing the efficacy of evaluating the gradients across different methodologies. 

n = 128
L = 30 
dx = L/n
x = torch.arange(-L/2, L/2, dx, dtype=float, requires_grad=True) #Independent
f = torch.cos(x) * torch.exp((-x**2)/25) #Dependent f(x)
df = -(torch.sin(x) * torch.exp((-x**2)/25) + (2/25)*x*f )#Derivative of f wrt x df_dx or D(f, x) -- Analytical Value


# x = torch.arange(-L/2, L/2, dx, dtype=float, requires_grad=True) #Independent
# f = torch.cos(2*x) + x**2 
# df = -2*torch.sin(2*x) + 2*x

# x = torch.arange(-L/2, L/2, dx, dtype=float, requires_grad=True) #Independent
# f = 3*x**2*torch.log(x)
# df = 6*x*torch.log(x) + 3*x**2/x 

df += x**2*df

# %% 
vars = sympy.symbols('f, x')
eqn_str = 'D(f, x) + x**2*D(f, x)' # Trying to evaluate the partial derivative of y wrt x

def deriv_AD(dep, ind):
     return torch.autograd.grad(dep,ind, grad_outputs=torch.ones_like(dep), create_graph=True, allow_unused=True)[0]

fn_AD = sympy.lambdify(vars, eqn_str, {'D': deriv_AD})
df_AD = fn_AD(f, x)

def deriv_FD(dep, ind):
    deriv = torch.zeros(dep.shape)
    deriv[:-1] = (dep[1:] - dep[:-1]) / dx
    deriv[-1] = deriv[-2]
    return deriv

fn_FD = sympy.lambdify(vars, eqn_str, {'D': deriv_FD})
df_FD = fn_FD(f, x)

# def deriv_FD_CS(dep, ind): #This is not CS but second order dummy !!
#     deriv = torch.zeros(dep.shape)
#     deriv[1:-1] = (dep[2:] -2*dep[1:-1] - dep[:-2]) / dx**2
#     deriv[-1] = deriv[-2]
#     deriv[0] = deriv[1]
#     return deriv

fn_FD_CS = sympy.lambdify(vars, eqn_str, {'D': deriv_FD})
df_FD_CS = fn_FD_CS(f, x)


# def deriv_Spectral(dep, ind):
#     kappa = torch.fft.fftfreq(len(dep), d=float(dx))
#     hat = torch.fft.fft(dep)
#     d_hat = (1j)*kappa*hat
#     return torch.fft.ifft(d_hat).real

def deriv_Spectral(dep, ind):
    fhat = torch.fft.fft(dep)
    kappa = (2*torch.pi/L)*torch.arange(-n/2,n/2)
    kappa = torch.fft.fftshift(kappa)
    dfhat = kappa*fhat*(1j)
    return torch.real(torch.fft.ifft(dfhat))


fn_Spectral = sympy.lambdify(vars, eqn_str, {'D': deriv_Spectral})
df_Spectral = fn_Spectral(f, x)

# %%

# plt.plot(x.detach().numpy(), f.detach().numpy(), label='Func')
plt.plot(x.detach().numpy(), df.detach().numpy(), lw=3.0,label='Analytical')
plt.plot(x.detach().numpy(), df_AD.detach().numpy(), '--', label='AD')
plt.plot(x.detach().numpy(), df_FD.detach().numpy(), '.-', label='FD')
plt.plot(x.detach().numpy(), df_FD_CS.detach().numpy(), ':', label='FD-CS')
plt.plot(x.detach().numpy(), df_Spectral.detach().numpy(), ':', label='Spectral')

plt.legend()
# %%
plt.plot(df_FD.detach() - df_FD_CS.detach())