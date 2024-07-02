#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24th May, 2024

Wrapper for Implementing Convolutional Operator as the Differential and Integral Operator 
- prefefined using a Finite Difference Scheme. 

Data used for all operations should be in the shape: BS, Nt, Nx, Ny
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


def pad_kernel(grid, kernel):#Could go into the deriv conv class
    kernel_size = kernel.shape[0]
    bs, nt, nx = grid.shape[0], grid.shape[1], grid.shape[2]
    return torch.nn.functional.pad(kernel, (0, nx - kernel_size, 0, nt-kernel_size), "constant", 0)


class ConvOperator():
    """
    A class for performing convolutions as a derivative or integrative operation on a given domain.
    By default the class instance evaluates the derivative

    Args:
        domain (str or tuple): The domain across which the derivative is taken.
            Can be 't' for time domain or ('x', 'y') for spatial domain.
        order (int): The order of derivation.
    """
    def __init__(self, domain=None, order=None, scale=1.0, taylor_order=2):

        try: 
            self.domain = domain #Axis across with the derivative is taken. 
            self.dims = len(self.domain) #Domain size
            self.order = order #order of derivation
            self.stencil = get_stencil(self.dims, self.order, taylor_order)

            if self.domain == 't':
                self.stencil = self.stencil
            elif self.domain == 'x':
                self.stencil = self.stencil.T
            elif self.domain == ('x','t'):
                self.stencil = self.stencil
            else:
                raise ValueError("Invalid Domain. Must be either x or t")
            
            self.kernel = self.stencil
            self.kernel = scale*self.kernel
        except:
            pass

    def convolution(self, field, kernel=None):
        """
        Performs 3D derivative convolution.

        Args:
            f (torch.Tensor): The input field tensor.
            k (torch.Tensor): The convolution kernel tensor.

        Returns:
            torch.Tensor: The result of the 3D derivative convolution.
        """
        if kernel != None: 
            self.kernel = kernel

        return F.conv2d(field.unsqueeze(1), self.kernel.unsqueeze(0).unsqueeze(0), padding=(self.kernel.shape[0]//2, self.kernel.shape[1]//2)).squeeze()
    

    def spectral_convolution(self, field, kernel=None):
        """
        Performs spectral convolution using the convolution theorem 

        f * g = \hat{f} . \hat{g}

        Args:
            f (torch.Tensor): The input field tensor.
            k (torch.Tensor): The convolution kernel tensor.

        Returns:
            torch.Tensor: The result of the 3D derivative convolution.
        """ 
        if kernel != None: 
            self.kernel = kernel

        field_fft = torch.fft.fftn(field, dim=(1,2))#t,x
        kernel_pad = pad_kernel(field, self.kernel)
        print(kernel_pad.shape)
        kernel_fft = torch.fft.fftn(kernel_pad)
        field_fft = torch.fft.fftn(field, dim=(1,2))#t,x,y

        return torch.fft.ifftn(field_fft * kernel_fft, dim=(1,2)).real


    def diff_integrate(self, field, kernel=None, eps=1e-6):

        """
        Performs Integration using the convolution theorem 3D derivative convolution.
        
        f * g * h = f  ; h =  1 / (g+eps)
        
        Args:
            field (torch.Tensor): The input field tensor.
            kernel (torch.Tensor): The convolution kernel tensor.
            eps (float): Avoiding NaNs
        Returns:
            torch.Tensor: The result of the 3D Integration Operation. 
        """
        
        if kernel != None: 
            self.kernel = kernel
        
        kernel_pad = pad_kernel(field, self.kernel)
        field_fft = torch.fft.fftn(field, dim=(1,2))#t,x
        kernel_fft = torch.fft.fftn(kernel_pad)
        inv_kernel_fft = 1 / (kernel_fft + eps)
        u_integrate = torch.fft.ifftn(field_fft * kernel_fft * inv_kernel_fft, dim=(1,2)).real
        return u_integrate


    def integrate(self, field, kernel=None, eps=1e-6):

        """
        Performs Integration using the convolution theorem 3D derivative convolution.
        
        f * g * h = f  ; h =  1 / (g+eps)
        
        Args:
            field (torch.Tensor): The input field tensor.
            kernel (torch.Tensor): The convolution kernel tensor.
            eps (float): Avoiding NaNs
        Returns:
            torch.Tensor: The result of the 3D Integration Operation. 
        """
        
        if kernel != None: 
            self.kernel = kernel
        
        kernel_pad = pad_kernel(field, self.kernel)
        field_fft = torch.fft.fftn(field, dim=(1,2))#t,x
        kernel_fft = torch.fft.fftn(kernel_pad)
        inv_kernel_fft = 1 / (kernel_fft + eps)
        u_integrate = torch.fft.ifftn(field_fft * inv_kernel_fft, dim=(1,2)).real
        return u_integrate
    


    def forward(self, field):
        """
        Performs the forward pass of the derivative convolution.

        Args:
            field (torch.Tensor): The input field tensor.

        Returns:
            torch.Tensor: The result of the derivative convolution.
        """
        return self.convolution(field, self.kernel)

    
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
    