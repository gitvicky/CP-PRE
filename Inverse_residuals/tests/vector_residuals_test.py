'''

Exploring vectors within the residual space

Inverse when the residual takes the form : x.D(y) = 0 


Testing with the equation: 

y = sin(x)
residual = x.D_x(y)

# In the Fourier space
# hat(x) * (hat(D)*hat(y)) = 0  

#Inverse ? 
# inverse = 1 / (hat(x) * (hat(D)*hat(y)) + eps)
'''

# %% 
import numpy as np
import torch 
import torch.nn.functional as F
from torch.fft import irfftn, rfftn
from matplotlib import pyplot as plt 

import sys
sys.path.append("/Users/Vicky/Documents/UKAEA/Code/Uncertainty_Quantification/PDE_Residuals")
from Utils.ConvOps_0d import ConvOperator


def differentiate(field, kernel=None, vector=None, correlation=False, slice_pad=True):
        """
        Performs custom differentiation using the convolution theorem.
        
        Args:
            field (torch.Tensor): Input tensor of shape (BS, Nt, Nx)
            kernel (torch.Tensor, optional): Optional custom kernel
            vector (torch.Tensor, optional): Optional vector

        Returns:
            torch.Tensor: Result of the differentiation operation
        """
        if kernel is not None:
            kernel = kernel
                # Add channel dimension for conv1d
        if field.dim() == 2:
            field = field.unsqueeze(1)
        

        pad_size = kernel.size(-1) // 2
        padded_field = F.pad(field, (pad_size,pad_size), mode='constant')
        field_fft = torch.fft.rfftn(padded_field.float(), dim=tuple(range(2, field.ndim)))
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        kernel_padding = [
            pad
            for i in reversed(range(2, padded_field.ndim))
            for pad in [0, padded_field.size(i) - kernel.size(i)]
        ]
        padded_kernel = F.pad(kernel, kernel_padding)

        kernel_fft = torch.fft.rfftn(padded_kernel.float(), dim = tuple(range(2, field.ndim)))

        if vector is None:
            vector_fft = torch.ones_like(field_fft)
        else: 
            vector = vector.unsqueeze(0)
            padded_vector = F.pad(vector, (pad_size,pad_size), mode='constant')
            vector_fft = torch.fft.rfftn(padded_vector.float(), dim = tuple(range(2, field.ndim)))

        if correlation==True:
            kernel_fft.imag *= -1

        output = irfftn(field_fft * kernel_fft * vector_fft, dim=tuple(range(2, field.ndim)))

        # Remove extra padded values
        if slice_pad==True:
            crop_slices = [slice(None), slice(None)] + [
                slice(0, (padded_field.size(i) - kernel.size(i) + 1), 1)#stride=1
                for i in range(2, padded_field.ndim)
            ]
            output = output[crop_slices].contiguous()

        return output.squeeze(1)

# %% 
x = torch.linspace(0, torch.pi, 100)
dx = x[1] - x[0]
y = torch.sin(x)
dy_dx = torch.cos(x)

D_x = ConvOperator(order=1)
residual = differentiate(y.unsqueeze(0), D_x.kernel, vector=x.unsqueeze(0), correlation=True)/(2*dx)

plt.figure(figsize=(10, 6))
plt.plot(x[1:-1], (x*dy_dx)[1:-1], 'b-', linewidth=2, label='x*dy_dx - actual')
plt.plot(x[1:-1], residual[0, 1:-1], 'r-', linewidth=2, label='x*dy_dx - convoluted')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# %% 


def integrate(field, kernel=None, vector=None, correlation=False, slice_pad=True):
        """
        Performs custom differentiation using the convolution theorem.
        
        Args:
            field (torch.Tensor): Input tensor of shape (BS, Nt, Nx)
            kernel (torch.Tensor, optional): Optional custom kernel
            vector (torch.Tensor, optional): Optional vector

        Returns:
            torch.Tensor: Result of the integration operation
        """
        if kernel is not None:
            kernel = kernel
                # Add channel dimension for conv1d
        if field.dim() == 2:
            field = field.unsqueeze(1)
        

        pad_size = kernel.size(-1) // 2
        padded_field = F.pad(field, (pad_size,pad_size), mode='constant')
        field_fft = torch.fft.rfftn(padded_field.float(), dim=tuple(range(2, field.ndim)))
        
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel_padding = [
            pad
            for i in reversed(range(2, padded_field.ndim))
            for pad in [0, padded_field.size(i) - kernel.size(i)]
        ]
        padded_kernel = F.pad(kernel, kernel_padding)

        kernel_fft = torch.fft.rfftn(padded_kernel.float(), dim = tuple(range(2, field.ndim)))

        if vector is None:
            vector_fft = torch.ones_like(field_fft)
        else: 
            vector = vector.unsqueeze(0)
            padded_vector = F.pad(vector, (pad_size+1,pad_size+1), mode='constant')
            vector_fft = torch.fft.rfftn(padded_vector.float(), dim = tuple(range(2, field.ndim)))

        if correlation==True:
            kernel_fft.imag *= -1

        inv_kernel = 1 / (kernel_fft * vector_fft + 1e-6)

        output = irfftn(field_fft * inv_kernel, dim=tuple(range(2, field.ndim)))

        # Remove extra padded values
        if slice_pad==True:
            crop_slices = [slice(None), slice(None)] + [
                slice(0, (padded_field.size(i) - kernel.size(i) + 1), 1)#stride=1
                for i in range(2, padded_field.ndim)
            ]
            output = output[crop_slices].contiguous()

        return output.squeeze(1)


retrieved = integrate(residual.unsqueeze(0), D_x.kernel, vector=None, correlation=True, slice_pad=True)*(2*dx)/x

plt.figure(figsize=(10, 6))
plt.plot(x[1:-1], (x*y)[1:-1], 'b-', linewidth=2, label='x*dy_dx - actual')
plt.plot(x[1:-1], retrieved[0, 1:-1], 'r-', linewidth=2, label='x*dy_dx - convoluted')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
# %%
