#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
22nd March, 2024

Implementing convolutions as Fourier Transform that relies on the Convolution Theorem. 
https://fkodom.substack.com/p/fourier-convolutions-in-pytorch
"""

# %% 
import numpy as np 
import torch 
import torch.nn.functional as F
from matplotlib import pyplot as plt 
# %% 
#Generating the signal - 2D Gaussian 
grid_size = 128
x = np.linspace(-1, 1, grid_size)
dx = x[1] -  x[0]
y = x.copy()
xx, yy = np.meshgrid(x, y)

uu = np.exp(-50*(xx**2 + yy**2))
signal = torch.Tensor(uu)

kernel_size = 3
kernel = torch.tensor([[0., 1., 0.],
                       [1., -4., 1.],
                       [0., 1., 0.]])



# signal  = signal.view(1, 1, grid_size, grid_size)
# kernel = kernel.view(1, 1, 3, 3)

# %% 
#Fourier Convolutions 

kernel_pad = F.pad(kernel, (0, grid_size - kernel_size, 0, grid_size - kernel_size), "constant", 0)

signal_fft = torch.fft.fft2(signal)
kernel_fft = torch.fft.fft2(kernel_pad)
dot_prod = signal_fft * kernel_fft
spectral_convolution = torch.fft.ifft2(dot_prod)

convolution = F.conv2d(signal.view(1, 1, grid_size, grid_size), kernel.view(1, 1, kernel_size, kernel_size), padding=(1,1))[0,0]

# %%
#Plotting the input, laplace, inverse 
from matplotlib import pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
fig = plt.figure(figsize=(20, 5))

mini = torch.min(convolution)
maxi = torch.max(convolution)

# Selecting the axis-X making the bottom and top axes False. 
plt.tick_params(axis='x', which='both', bottom=False, 
                top=False, labelbottom=False) 
  
# Selecting the axis-Y making the right and left axes False 
plt.tick_params(axis='y', which='both', right=False, 
                left=False, labelleft=False) 
  # Remove frame
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)


ax = fig.add_subplot(1,4,1)
pcm =ax.imshow(uu, cmap=cm.coolwarm)#, vmin=mini, vmax=maxi)
ax.title.set_text('Input')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(1,4,2)
pcm =ax.imshow(convolution, cmap=cm.coolwarm,  vmin=mini, vmax=maxi)
ax.title.set_text('Convolution')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(1,4,3)
pcm =ax.imshow(spectral_convolution.real, cmap=cm.coolwarm, vmin=mini, vmax=maxi)
ax.title.set_text('FFT Conv.')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(1,4,4)
pcm =ax.imshow(spectral_convolution.real - convolution, cmap=cm.coolwarm)#, vmin=mini, vmax=maxi)
ax.title.set_text('Conv Error.')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)


# %%
#Finding the inverse Kernel 
eps = 1e-16
inv_kernel_fft = 1 / (kernel_fft + eps)
inv_kernel = torch.fft.ifft2(inv_kernel_fft).real
uu_retrieve = torch.fft.ifft2(signal_fft * kernel_fft * inv_kernel_fft).real

# %%
#Plotting the input, laplace, inverse 
from matplotlib import pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
fig = plt.figure(figsize=(8, 8))

# Selecting the axis-X making the bottom and top axes False. 
plt.tick_params(axis='x', which='both', bottom=False, 
                top=False, labelbottom=False) 
  
# Selecting the axis-Y making the right and left axes False 
plt.tick_params(axis='y', which='both', right=False, 
                left=False, labelleft=False) 
  # Remove frame
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)


ax = fig.add_subplot(2,2,1)
pcm =ax.imshow(uu, cmap=cm.coolwarm)#, vmin=mini, vmax=maxi)
ax.title.set_text(r'$f$')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(2,2,2)
pcm =ax.imshow(convolution, cmap=cm.coolwarm)#,  vmin=mini, vmax=maxi)
ax.title.set_text(r'$f*k$')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(2,2,3)
pcm =ax.imshow(spectral_convolution.real, cmap=cm.coolwarm)#, vmin=mini, vmax=maxi)
ax.title.set_text(r'$F^{-1} (\hat{f} \cdot \hat{k})$')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(2,2,4)
pcm =ax.imshow(uu_retrieve, cmap=cm.coolwarm)#, vmin=mini, vmax=maxi)
ax.title.set_text(r'$F^{-1} (\hat{f} \cdot \hat{k} \cdot (1/\hat{k}))$')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)


# %%
