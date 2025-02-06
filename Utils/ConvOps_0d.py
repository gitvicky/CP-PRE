#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper for Implementing Convolutional Operator as the Differential and Integral Operator
for ODEs using Finite Difference Schemes.

Data used for all operations should be in the shape: BS, Nt

For computational simplicity in backpropagation, the convolution is implemented as a correlation 
correlation: complex conjuateof the convolution operator in the frequency space. 
This is explained in :
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8721631

fftconv takes care of this automatically, within diff and integrate its an argument
that allows us to choose between a convolution and a correlation. 
"""
import torch
import torch.nn.functional as F
from fft_conv_pytorch.fft_conv import * 

def get_stencil(deriv_order, taylor_order=2):
    """
    Returns the finite difference stencil for temporal derivatives.
    
    Args:
        deriv_order (int): Order of the derivative (1 or 2)
        taylor_order (int): Order of accuracy for Taylor expansion (2, 4, or 6)
    
    Returns:
        torch.Tensor: The finite difference stencil
    """

    if deriv_order == 0:  # Identity convolution
        return torch.tensor([0., 1., 0.], dtype=torch.float32)
    elif deriv_order == 2 and taylor_order == 2:
        return torch.tensor([1., -2., 1.], dtype=torch.float32)
    elif deriv_order == 1 and taylor_order == 2:
        return torch.tensor([-1., 0., 1.], dtype=torch.float32)
    elif deriv_order == 2 and taylor_order == 4:
        return torch.tensor([-1/12, 4/3, -5/2, 4/3, -1/12], dtype=torch.float32)
    elif deriv_order == 2 and taylor_order == 6:
        return torch.tensor([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90], dtype=torch.float32)
    elif deriv_order == 1 and taylor_order == 4:
        return torch.tensor([1/12, -2/3, 0, 2/3, -1/12], dtype=torch.float32)
    
    raise ValueError("Invalid stencil parameters")



class ConvOperator():
    """
    A class for performing convolutions as a derivative or integrative operation
    on temporal data for ODEs.

    Args:
        order (int): The order of derivation (1 or 2)
        scale (float): Scaling factor for the kernel
        taylor_order (int): Order of accuracy for Taylor expansion
        conv (str): Convolution method ('direct' or 'spectral')
        device (str): Device to use for computations ('cpu' or 'cuda')
        requires_grad (bool): Whether the kernel should require gradients
    """
    def __init__(self, order=None, scale=1.0, taylor_order=2, conv='direct', 
                 device='cpu', requires_grad=False):
        try:
            self.order = order
            self.stencil = get_stencil(self.order, taylor_order)
            self.kernel = (scale * self.stencil).to(device)

            if requires_grad:
                self.kernel.requires_grad_ = True

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
        Performs 1D temporal derivative convolution.

        Args:
            field (torch.Tensor): Input tensor of shape (BS, Nt)
            kernel (torch.Tensor, optional): Optional custom kernel

        Returns:
            torch.Tensor: Result of the temporal convolution
        """
        if kernel is not None:
            self.kernel = kernel

        # Add channel dimension for conv1d
        if field.dim() == 2:
            field = field.unsqueeze(1)
        kernel = self.kernel.unsqueeze(0).unsqueeze(0)

        convdirect = F.conv1d(field, kernel, padding=self.kernel.shape[-1]//2).squeeze(1)

        return convdirect
    

    def spectral_convolution(self, field, kernel=None):
        """
        Performs spectral convolution using the convolution theorem.

        Args:
            field (torch.Tensor): Input tensor of shape (BS, Nt)
            kernel (torch.Tensor, optional): Optional custom kernel

        Returns:
            torch.Tensor: Result of the spectral convolution
        """
        if kernel is not None:
            self.kernel = kernel

        # Add channel dimension for conv1d
        if field.dim() == 2:
            field = field.unsqueeze(1)
        kernel = self.kernel.unsqueeze(0).unsqueeze(0)
        convfft = fft_conv(field, kernel, padding=self.kernel.shape[-1]//2).squeeze(1)

        return convfft


    def differentiate(self, field, kernel=None, eps=1e-6, correlation=False):
        """
        Performs integration using the convolution theorem.
        
        Args:
            field (torch.Tensor): Input tensor of shape (BS, Nt)
            kernel (torch.Tensor, optional): Optional custom kernel
            eps (float): Small value to avoid division by zero

        Returns:
            torch.Tensor: Result of the integration operation
        """
        if kernel is not None:
            self.kernel = kernel

                # Add channel dimension for conv1d
        if field.dim() == 2:
            field = field.unsqueeze(1)

        pad_size = self.kernel.size(-1) // 2
        padded_field = F.pad(field, (pad_size,pad_size), mode='constant')
        field_fft = torch.fft.rfftn(padded_field, dim=tuple(range(2, field.ndim)))
        kernel = self.kernel.unsqueeze(0).unsqueeze(0)


        kernel_padding = [
            pad
            for i in reversed(range(2, padded_field.ndim))
            for pad in [0, padded_field.size(i) - kernel.size(i)]
        ]
        padded_kernel = f.pad(kernel, kernel_padding)

        kernel_fft = torch.fft.rfftn(padded_kernel, dim = tuple(range(2, field.ndim)))
        if correlation==True:
            kernel_fft.imag *= -1

        output = irfftn(field_fft * kernel_fft, dim=tuple(range(2, field.ndim))).squeeze(1)
        return output
  
    
    def integrate(self, field, kernel=None, correlation=False, eps=1e-6):
        """
        Performs direct integration in the frequency domain.
        
        Args:
            field (torch.Tensor): Input tensor of shape (BS, Nt)
            kernel (torch.Tensor, optional): Optional custom kernel
            eps (float): Small value to avoid division by zero

        Returns:
            torch.Tensor: Result of the integration operation
        """
        if kernel is not None:
                self.kernel = kernel

        # Add channel dimension for conv1d
        if field.dim() == 2:
            field = field.unsqueeze(1)

        pad_size = self.kernel.size(-1) // 2
        padded_field = F.pad(field, (pad_size,pad_size), mode='constant')
        padded_field = field

        field_fft = torch.fft.rfftn(padded_field, dim=tuple(range(2, field.ndim)))
        kernel = self.kernel.unsqueeze(0).unsqueeze(0)

        kernel_padding = [
            pad
            for i in reversed(range(2, padded_field.ndim))
            for pad in [0, padded_field.size(i) - kernel.size(i)]
        ]
        padded_kernel = f.pad(kernel, kernel_padding)

        kernel_fft = torch.fft.rfftn(padded_kernel, dim = tuple(range(2, field.ndim)))
        # kernel_fft.imag *= -1

        inv_kernel_fft = 1 / (kernel_fft + eps)

        if correlation == True:
            inv_kernel_fft.imag *= -1 

        output = irfftn(field_fft * inv_kernel_fft, dim=tuple(range(2, field.ndim))).squeeze(1)
        return output


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
# #Example Usage 
# from matplotlib import pyplot as plt 

# # Random Signal
# signal = torch.randn(1,100)  # (batch_size, channels, signal_length)

# D_tt = ConvOperator(order=2, taylor_order=2, conv='direct')
# directconv = D_tt(signal)

# D_tt = ConvOperator(order=2, taylor_order=2, conv='spectral')
# fftconv = D_tt(signal)

# mine = D_tt.differentiate(signal)

# plt.figure()
# plt.plot(fftconv[0], label='spectral')
# plt.plot(directconv[0], label='direct')
# plt.plot(mine[0], label='mine')
# plt.legend()

# # %% 
# retrieved_sgnal= D_tt.integrate(mine)

# plt.figure()
# plt.plot(signal[0], label='actual')
# plt.plot(retrieved_sgnal[0,1:-1], label='retreievd')
# plt.legend()
# # %%
