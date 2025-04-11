# %% 
import torch.nn.functional as F
from torch.fft import irfftn, rfftn
import numpy as np
import matplotlib.pyplot as plt

#y = sin(x)
#Case we are interested in is x*dy_dx = cos(x), where dy_dx should be evaluated using a 
# convolutional kernel and we want to be able to invert the expression to obtain x*y

x_vals = np.linspace(0.01, np.pi, 100)
y_vals = np.sin(x_vals)
y_deriv = np.cos(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, x_vals*y_vals, 'b-', linewidth=2, label='x*y')
plt.plot(x_vals, x_vals*y_deriv, 'b-', linewidth=2, label='x*dy_dx')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# %%
import torch 
# Import your ConvOps module
import sys
sys.path.append("/Users/Vicky/Documents/UKAEA/Code/Uncertainty_Quantification/PDE_Residuals")
from Utils.ConvOps_0d import ConvOperator

x = torch.tensor(x_vals, dtype=torch.float32).unsqueeze(0)
y = torch.tensor(y_vals, dtype=torch.float32).unsqueeze(0)
y_x = torch.tensor(y_deriv, dtype=torch.float32).unsqueeze(0)

dx = x_vals[1]-x_vals[0]
D_x = ConvOperator(order=1, conv='spectral')#, scale=alpha)

D_identity = ConvOperator(order=0) #Identity 
D_identity.kernel = torch.tensor([0.0, 1.0, 0.0])

residual = D_x.differentiate(y, D_x.kernel, correlation=True)/(2*dx)

plt.figure(figsize=(10, 6))
plt.plot(x_vals[1:-1], y_deriv[1:-1], 'b-', linewidth=2, label='x*dy_dx - actual')
plt.plot(x_vals[1:-1], residual[0, 1:-1], 'r-', linewidth=2, label='x*dy_dx - convoluted')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# %% 
residual = D_x.differentiate(y, D_x.kernel, correlation=True, slice_pad=False)/(2*dx)
retrieved = D_x.integrate(residual, D_x.kernel, correlation=True, slice_pad=True)*(2*dx)

plt.figure(figsize=(10, 6))
plt.plot(x_vals[1:-1], y_vals[1:-1], 'b-', linewidth=2, label='y - actual')
plt.plot(x_vals[1:-1], retrieved[0, 1:-1], 'r-', linewidth=2, label='y - retrieved')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)


# %% 
#Handling vectors in the kernels 

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

residual = differentiate(y, D_x.kernel, vector=None, correlation=True)/(2*dx)

plt.figure(figsize=(10, 6))
plt.plot(x_vals[1:-1], y_deriv[1:-1], 'b-', linewidth=2, label='x*dy_dx - actual')
plt.plot(x_vals[1:-1], residual[0, 1:-1], 'r-', linewidth=2, label='x*dy_dx - convoluted')
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

residual = differentiate(y, D_x.kernel, correlation=True, slice_pad=False)/(2*dx)
retrieved = integrate(residual, D_x.kernel, vector=x, correlation=True, slice_pad=True)*(2*dx)

plt.figure(figsize=(10, 6))
plt.plot(x_vals[1:-1], x_vals[1:-1]*y_vals[1:-1], 'b-', linewidth=2, label='x*dy_dx - actual')
plt.plot(x_vals[1:-1], retrieved[0, 2:-2], 'r-', linewidth=2, label='x*dy_dx - convoluted')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
# %%
