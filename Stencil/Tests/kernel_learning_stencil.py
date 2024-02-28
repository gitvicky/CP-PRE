#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
28th Feb, 2023

Learning the required Convolutional Kernels 
    1. Try and learn the finite difference stencil for performing the residual estimation
    2. If we are able to retrieve the FD stencils, attempt tp learnt the TranspConv kernel that maps from the residual to the field. 

Using the wave equation as the base test case for this experiment.
"""

# %% 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm 
import torch 
import torch.nn.functional as F
from pyDOE import lhs
# %% 
#Generating the datasets: Performing the Laplace Operator over a range of ICs. 

def generate_input_data(N):

    x = np.linspace(1, -1, 31)
    y = x.copy()
    xx, yy = np.meshgrid(x, y)

    lb = np.asarray([-10, 0.10, 0.10]) #Amp, x-pos, y-pos 
    ub = np.asarray([-50, 0.70, 0.70]) #Amp, x-pos, y-pos
    
    
    params = lb + (ub-lb)*lhs(3, N)
    amp, x_pos, y_pos = params[:, 0], params[:, 1], params[:, 2]
    
    u_ic = []
    for ii in range(N):
        u_ic.append(np.exp(amp[ii]*((xx-x_pos[ii])**2 + (yy-y_pos[ii])**2)))
    return np.asarray(u_ic)

uu = generate_input_data(10)
uu = torch.Tensor(uu)

# %% 
#Obtaining the laplacianm of the field evaluated using the 5-point stencil. 
def generate_output_data(X):    
    laplace_kernel = torch.tensor([[0., 1., 0.],
                           [1., -4., 1.],
                           [0, 1., 0.]])
    
    conv = F.conv2d(X.view(X.shape[0], 1, X.shape[1], X.shape[2]), laplace_kernel.view(1,1,3,3))

uu_laplace = generate_output_data(uu)

