#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24th May, 2024

Wrapper for Implementing Convolutional Operator as the Differential Operator with a Finite Difference Scheme. 

"""
import numpy as np 
import torch 
import torch.nn.functional as F


def get_stencil(dims, deriv_order, taylor_order=2):

    if dims == 1:
        if deriv_order == 2 and taylor_order == 2:
            return torch.tensor([
                [0, 1, 0],
                [0, -2, 0],
                [0, 1, 0]
            ], dtype=torch.float32)
        elif deriv_order == 1 and taylor_order == 2:
            return torch.tensor([
                [0, -1, 0],
                [0, 0, 0],
                [0, 1, 0]
            ], dtype=torch.float32)
    elif dims == 2:
        if deriv_order == 2 and taylor_order == 2:
            return torch.tensor([
                [0., 1., 0.],
                [1., -4., 1.],
                [0., 1., 0.]
            ], dtype=torch.float32)
        elif deriv_order == 2 and taylor_order == 4:
            return torch.tensor([
                [0, 0, -1/12, 0, 0],
                [0, 0, 4/3, 0, 0],
                [-1/12, 4/3, -5/2, 4/3, -1/12],
                [0, 0, 4/3, 0, 0],
                [0, 0, -1/12, 0, 0]
            ], dtype=torch.float32)
        elif deriv_order == 2 and taylor_order == 6:
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
    else:
        raise ValueError("Invalid axis. Must be either 0, 1 or 2")
    
    return kernel


class DerivConv():
    """
    A class for performing derivative convolutions on a given domain.

    Args:
        domain (str or tuple): The domain across which the derivative is taken.
            Can be 't' for time domain or ('x', 'y') for spatial domain.
        order (int): The order of derivation.
    """
    def __init__(self, domain, order, taylor_order=2):
        self.domain = domain #Axis across with the derivative is taken. 
        self.dims = len(self.domain) #Domain size
        self.order = order #order of derivation
        self.stencil = get_stencil(self.dims, self.order, taylor_order)

        if self.domain == 't':
            self.axis = 2
        elif self.domain == 'x':
            self.axis = 0
        elif self.domain == 'y':
            self.axis = 1
        elif self.domain == ('x','y'):
            self.axis = 0
        elif self.domain == ('x', 'y', 't'):
            self.axis = 0
        else:
            raise ValueError("Invalid Domain. Must be either x,y or t")
        
        self.kernel = kernel_3d(self.stencil, self.axis)


    def conv_deriv_3d(self, f, k):
        """
        Performs 3D derivative convolution.

        Args:
            f (torch.Tensor): The input field tensor.
            k (torch.Tensor): The convolution kernel tensor.

        Returns:
            torch.Tensor: The result of the 3D derivative convolution.
        """
        return F.conv3d(f.unsqueeze(1), k.unsqueeze(0).unsqueeze(0), padding=(k.shape[0]//2, k.shape[1]//2, k.shape[2]//2)).squeeze()

    
    def forward(self, field):
        """
        Performs the forward pass of the derivative convolution.

        Args:
            field (torch.Tensor): The input field tensor.

        Returns:
            torch.Tensor: The result of the derivative convolution.
        """
        return self.conv_deriv_3d(field, self.kernel)

    
    def __call__(self, inputs):
        """
        Performs the forward pass computation when the instance is called.

        Args:
            inputs (torch.Tensor): The input tensor to the derivative convolution. 
            -has to be in shape (BS, Nt, Nx, Ny)

        Returns:
            torch.Tensor: The result of the derivative convolution.
        """
        outputs = self.forward(inputs)
        return outputs
    

