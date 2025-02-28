#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24th May, 2024

Wrapper for Implementing Convolutional Operator as the Differential and Integral Operator 
- prefefined using a Finite Difference Scheme. 

Data used for all operations should be in the shape: BS, Nt, Nx
"""
import numpy as np 
import torch 
import torch.nn.functional as F
from fft_conv_pytorch.fft_conv import * 


def get_stencil(dims, deriv_order, taylor_order=2):

    if dims == 1:
        if deriv_order == 0:  # Identity convolution
            return torch.tensor([
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]
            ], dtype=torch.float32)
        elif deriv_order == 2 and taylor_order == 2:
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
    def __init__(self, domain=None, order=None, scale=1.0, taylor_order=2, conv='direct', device='cpu'):

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
            self.kernel = self.kernel.to(device)
        except:
            pass

        if conv == 'direct': 
            self.conv = self.convolution
        elif conv == 'spectral':
            self.conv = self.spectral_convolution
        else:
            raise ValueError("Unknown Convolution Method")
        

    def convolution(self, field, kernel=None):
        """
        Performs 2D derivative convolution.

        Args:
            f (torch.Tensor): The input field tensor of shape (BS, Nt, Nx)
            k (torch.Tensor): The convolution kernel tensor.

        Returns:
            torch.Tensor: The result of the 2D derivative convolution.
        """
        if kernel != None: 
            self.kernel = kernel

        # Add channel dimension for conv2d
        if field.dim() == 3:
            field = field.unsqueeze(1)

        kernel = self.kernel.unsqueeze(0).unsqueeze(0)

        return F.conv2d(field, kernel, padding=(self.kernel.shape[0]//2, self.kernel.shape[1]//2)).squeeze(1)
    

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

        # Add channel dimension for conv2d
        if field.dim() == 3:
            field = field.unsqueeze(1)
        kernel = self.kernel.unsqueeze(0).unsqueeze(0)
        convfft = fft_conv(field, kernel, padding=(self.kernel.shape[0]//2, self.kernel.shape[1]//2)).squeeze(1)

        return convfft
    

    def differentiate(self, field, kernel=None, correlation=False, slice_pad=True):

        """
        Performs custom differentiation using the convolution theorem.
        
        Args:
            field (torch.Tensor): Input tensor of shape (BS, Nt, Nx)
            kernel (torch.Tensor, optional): Optional custom kernel

        Returns:
            torch.Tensor: Result of the differentiation operation
        """

        if kernel is not None:
            self.kernel = kernel

        # Add channel dimension for conv2d
        if field.dim() == 3:
            field = field.unsqueeze(1)

        pad_size = self.kernel.size(-1) // 2
        padded_field = F.pad(field, (pad_size, pad_size, pad_size, pad_size), mode='constant')
        
        field_fft = torch.fft.rfftn(padded_field.float(), dim=tuple(range(2, field.ndim)))
        kernel = self.kernel.unsqueeze(0).unsqueeze(0)

        kernel_padding = [
            pad
            for i in reversed(range(2, padded_field.ndim))
            for pad in [0, padded_field.size(i) - kernel.size(i)]
        ]
        padded_kernel = F.pad(kernel, kernel_padding)

        kernel_fft = torch.fft.rfftn(padded_kernel.float(), dim = tuple(range(2, field.ndim)))
        
        if correlation == True:
            kernel_fft.imag *= -1

        output = irfftn(field_fft * kernel_fft, dim=tuple(range(2, field.ndim)))
        
        # Remove extra padded values
        if slice_pad == True:
            crop_slices = [slice(None), slice(None)] + [
                slice(0, (padded_field.size(i) - kernel.size(i) + 1), 1)#stride=1
                for i in range(2, padded_field.ndim)
            ]
            output = output[crop_slices].contiguous()

        return output.squeeze(1)
    

    def integrate(self, field, kernel=None, correlation=False, slice_pad=True, eps=1e-6):
        """
        Performs direct integration in the frequency domain.

                f * g * h = f  ; h =  1 / (g+eps)
        
        Args:
            field (torch.Tensor): Input tensor of shape (BS, Nt, Nx)
            kernel (torch.Tensor, optional): Optional custom kernel
            eps (float): Small value to avoid division by zero

        Returns:
            torch.Tensor: Result of the integration operation
        """
        if kernel is not None:
                self.kernel = kernel

        # Add channel dimension for conv2d
        if field.dim() == 3:
            field = field.unsqueeze(1)

        pad_size = self.kernel.size(-1) // 2
        padded_field = F.pad(field, (pad_size, pad_size, pad_size, pad_size), mode='constant')
        # padded_field = field

        field_fft = torch.fft.rfftn(padded_field, dim=tuple(range(2, field.ndim)))
        kernel = self.kernel.unsqueeze(0).unsqueeze(0)
        

        kernel_padding = [
            pad
            for i in reversed(range(2, padded_field.ndim))
            for pad in [0, padded_field.size(i) - kernel.size(i)]
        ]
        padded_kernel = F.pad(kernel, kernel_padding)

        kernel_fft = torch.fft.rfftn(padded_kernel, dim = tuple(range(2, field.ndim)))
        # kernel_fft.imag *= -1

        inv_kernel_fft = 1 / (kernel_fft + eps)

        if correlation == True:
            inv_kernel_fft.imag *= -1 

        output = irfftn(field_fft * inv_kernel_fft, dim=tuple(range(2, field.ndim)))

            # Remove extra padded values
        if slice_pad == True:
            crop_slices = [slice(None), slice(None)] + [
                slice(0, (padded_field.size(i) - kernel.size(i) + 1), 1)#stride=1
                for i in range(2, padded_field.ndim)
            ]

            output = output[crop_slices].contiguous()

        return output.squeeze(1)
    


    def forward(self, field):
        """
        Performs the forward pass of the derivative convolution.

        Args:
            field (torch.Tensor): Input tensor of shape (BS, Nt)

        Returns:
            torch.Tensor: Result of the derivative convolution
        """
        return self.conv(field, self.kernel)

    def __call__(self, inputs):
        """
        Callable interface for the ConvOperator.

        Args:
            inputs (torch.Tensor): Input tensor of shape (BS, Nt)

        Returns:
            torch.Tensor: Result of the derivative convolution
        """
        return self.forward(inputs)

# %% 
#Example Usage
import torch 
from matplotlib import pyplot as plt 

def convection_solution(initial_condition, c, dt, nt):
    """
    Implementation of convection equation with analytical solution u(x,t) = f(x - ct)
    
    Args:
        initial_condition: Tensor of shape [1, 1, Nx] containing the initial condition
        c: Convection velocity (scalar)
        dt: Time step
        nt: Number of time steps
    
    Returns:
        Tensor of shape [1, Nt, Nx] containing the solution
    """
    return torch.cat([initial_condition(torch.arange(initial_condition.shape[2]).unsqueeze(0) - c * i * dt) for i in range(nt)], dim=1)


# Define parameters
nx = 100
nt = 50
c = 1.0
dt = 0.1
x = np.linspace(0, 1 ,nx)
dx = x[1]-x[0]

# Initial condition function (Gaussian)
def initial_condition(x):
    return torch.exp(-(x - nx/2)**2 / 50).reshape(1, 1, -1)

# One-liner solution
solution = torch.cat([initial_condition(torch.arange(nx).unsqueeze(0) - c * i * dt) for i in range(nt)], dim=1)
# plt.plot(solution[0].T)


# %%

D_t = ConvOperator(domain='t', order=1)
D_x = ConvOperator(domain='x', order=1)
D = ConvOperator()
D.kernel = D_t.kernel + dt/dx*c*D_x.kernel

direct_res = D(solution) #direct convolution
spectral_res = D.spectral_convolution(solution) #spectral convolution
manual_res = D.differentiate(solution, correlation=True, slice_pad=True) #Manual

# %%
#Inverse 
diff = D.differentiate(solution, correlation=False, slice_pad=False)
integ = D.integrate(diff, correlation=True, slice_pad=True)
# %%
