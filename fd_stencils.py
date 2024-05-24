#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
22nd April, 2024

Finite Difference Stencils

"""
import numpy as np 
import torch 
import torch.nn.functional as F

# Dimensionality,  Number of Points, Order of differentiation, Order of Approximation of the Taylor Series 

def get_stencil(dims, points, deriv_order, taylor_order=2):

    if dims == 1:
        if points == 3 and deriv_order == 2 and taylor_order == 2:
            return torch.tensor([
                [0, 1, 0],
                [0, -2, 0],
                [0, 1, 0]
            ], dtype=torch.float32)
        elif points == 2 and deriv_order == 1 and taylor_order == 2:
            return torch.tensor([
                [0, -1, 0],
                [0, 0, 0],
                [0, 1, 0]
            ], dtype=torch.float32)
    elif dims == 2:
        if points == 5 and deriv_order == 2 and taylor_order == 2:
            return torch.tensor([
                [0., 1., 0.],
                [1., -4., 1.],
                [0., 1., 0.]
            ], dtype=torch.float32)
        elif points == 9 and deriv_order == 2 and taylor_order == 4:
            return torch.tensor([
                [0, 0, -1/12, 0, 0],
                [0, 0, 4/3, 0, 0],
                [-1/12, 4/3, -5/2, 4/3, -1/12],
                [0, 0, 4/3, 0, 0],
                [0, 0, -1/12, 0, 0]
            ], dtype=torch.float32)
        elif points == 15 and deriv_order == 2 and taylor_order == 6:
            return torch.tensor([
                [0, 0, 0, 1/90, 0, 0, 0],
                [0, 0, 0, -3/20, 0, 0, 0],
                [0, 0, 0, 3/2, 0, 0, 0],
                [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90],
                [0, 0, 0, 3/2, 0, 0, 0],
                [0, 0, 0, -3/20, 0, 0, 0],
                [0, 0, 0, 1/90, 0, 0, 0]
            ], dtype=torch.float32)

    raise ValueError("Invalid stencil parameters")

#If the data is BS, Nt, Nx, Ny -- then the axis=0,1 will be for spatial derivs and axis=2 wil be for time. 
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

