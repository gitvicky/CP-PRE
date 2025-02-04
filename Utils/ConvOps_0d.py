#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper for Implementing Convolutional Operator as the Differential and Integral Operator
for ODEs using Finite Difference Schemes.

Data used for all operations should be in the shape: BS, Nt
"""
import torch
import torch.nn.functional as F


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


def pad_kernel(grid, kernel):
    """Pads the kernel to match the temporal dimension of the input."""
    kernel_size = kernel.shape[0]
    bs, nt = grid.shape[0], grid.shape[1]
    return F.pad(kernel, (0, nt - kernel_size), "constant", 0)


class ConvOperator():
    """
    A class for performing convolutions as a derivative or integrative operation
    on temporal data for ODEs.

    Args:
        order (int): The order of derivation (1 or 2)
        scale (float): Scaling factor for the kernel
        taylor_order (int): Order of accuracy for Taylor expansion
        conv (str): Convolution method ('conv' or 'spectral')
        device (str): Device to use for computations ('cpu' or 'cuda')
        requires_grad (bool): Whether the kernel should require gradients
    """
    def __init__(self, order=None, scale=1.0, taylor_order=2, conv='conv', 
                 device='cpu', requires_grad=False):
        try:
            self.order = order
            self.stencil = get_stencil(self.order, taylor_order)
            self.kernel = (scale * self.stencil).to(device)

            if requires_grad:
                self.kernel.requires_grad_ = True

        except:
            pass

        if conv == 'conv':
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
        field = field.unsqueeze(1)
        kernel = self.kernel.unsqueeze(0).unsqueeze(0)
        
        # Perform convolution with appropriate padding
        padding = self.kernel.shape[0] // 2
        return F.conv1d(field, kernel, padding=padding).squeeze(1)

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

        field_fft = torch.fft.fft(field, dim=1)
        kernel_pad = pad_kernel(field, self.kernel)
        kernel_fft = torch.fft.fft(kernel_pad)

        return torch.fft.ifft(field_fft * kernel_fft, dim=1).real

    def diff_integrate(self, field, kernel=None, eps=1e-6):
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
        
        field_fft = torch.fft.fft(field, dim=1)
        kernel_pad = pad_kernel(field, self.kernel)
        kernel_fft = torch.fft.fft(kernel_pad)
        inv_kernel_fft = 1 / (kernel_fft + eps)
        
        return torch.fft.ifft(field_fft * kernel_fft * inv_kernel_fft, dim=1).real

    def integrate(self, field, kernel=None, eps=1e-6):
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
        
        field_fft = torch.fft.fft(field, dim=1)
        kernel_pad = pad_kernel(field, self.kernel)
        kernel_fft = torch.fft.fft(kernel_pad)
        inv_kernel_fft = 1 / (kernel_fft + eps)
        
        return torch.fft.ifft(field_fft * inv_kernel_fft, dim=1).real

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