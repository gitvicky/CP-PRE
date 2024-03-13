
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7th March, 2024

Finite Difference as Matrix Multiplication. 

If the Stencil (K) can be written in its Matrix Form as W. 
Field Matrix flattened out as the fwd_laplace = X 
The residual (laplacian in this case) fwd_laplace_soln = Y = W X
The inverse of the laplace inv_laplace = W^-1
Retreiving the input from the laplace inv_laplace_soln = W^-1 Y = W^-1 W X
"""
# %% 

import numpy as np

def finite_difference_matrix_2d(nx, ny, stencil, stencil_center):
    """
    Constructs the finite difference matrix for a given 2D stencil and grid sizes.

    Args:
        nx (int): The number of grid points in the x-direction.
        ny (int): The number of grid points in the y-direction.
        stencil (np.ndarray): A 2D array representing the finite difference stencil.
        stencil_center (tuple): The center of the stencil, given as (x_center, y_center).

    Returns:
        np.ndarray: The finite difference matrix.
    """
    kx, ky = stencil.shape  # Order of the stencil in x and y directions
    n = nx * ny  # Total number of grid points
    A = np.zeros((n, n))

    x_center, y_center = stencil_center  # Center of the stencil

    # Fill the matrix rows
    for i in range(nx):
        for j in range(ny):
            row_idx = i * ny + j  # Linear index for the current grid point
            for kx_idx in range(kx):
                for ky_idx in range(ky):
                    # Compute the neighbor indices
                    ix = i + kx_idx - x_center
                    jy = j + ky_idx - y_center

                    # Skip out-of-bounds neighbors
                    if ix < 0 or ix >= nx or jy < 0 or jy >= ny:
                        continue

                    col_idx = ix * ny + jy  # Linear index for the neighbor
                    A[row_idx, col_idx] = stencil[kx_idx, ky_idx]

    return A



# %% 
grid_size = 128
x = np.linspace(-1, 1, grid_size) #gridsize
dx = x[1]-x[0]
y = x.copy()
xx, yy = np.meshgrid(x, y)
uu = X = np.exp(-20 *(xx**2 + yy**2)) #2D Gaussian Initial Condition. 


stencil = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]])  # 2D Laplacian stencil-

stencil_center = (1, 1)  # Center of the stencil
nx, ny = grid_size, grid_size
import tracemalloc
tracemalloc.start()

fwd_laplace = W = finite_difference_matrix_2d(nx, ny, stencil, stencil_center) #fwd_laplace
inv_laplace = np.linalg.inv(W)

# %%
#Inverting a block diagonal matrix -- look into the scipy sparse matrix library
#https://docs.scipy.org/doc/scipy/reference/sparse.html

# %%
fwd_laplace_soln = Y = np.matmul(W, uu.reshape(-1)).reshape(nx, ny)
inv_laplace_soln = X_ = np.matmul(inv_laplace, Y.reshape(-1)).reshape(nx, ny)
# %%
#Plotting the input, laplace, inverse 
from matplotlib import pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
fig = plt.figure(figsize=(20, 5))

mini = np.min(X)
maxi = np.max(X)


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


ax = fig.add_subplot(1,3,1)
pcm =ax.imshow(X, cmap=cm.coolwarm)#, vmin=mini, vmax=maxi)
ax.title.set_text('Input')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(1,3,2)
pcm =ax.imshow(fwd_laplace_soln, cmap=cm.coolwarm)#,  vmin=mini, vmax=maxi)
ax.title.set_text('Forward Laplace')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(1,3,3)
pcm =ax.imshow(inv_laplace_soln, cmap=cm.coolwarm, vmin=mini, vmax=maxi)
ax.title.set_text('Inverse Laplace')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

# %%
#Cholesky Decomposition of the Stencil Matrix 

# A = np.array([[4, 1, 1],
#               [1, 2, 3],
#               [1, 3, 6]])

A = W #weight matrix
def cholesky_inverse(A):
    # Perform Cholesky decomposition
    L = np.linalg.cholesky(A)

    # Compute the inverse of L
    L_inv = np.linalg.inv(L)

    # # Compute the inverse of L^T
    # L_T_inv = np.linalg.inv(L.T)
    L_T_inv = L_inv.T

    # Compute the inverse of A
    A_inv = np.dot(L_T_inv, L_inv)

    return A_inv

# %timeit A_inverse = np.linalg.inv(A)
# %timeit A_cholesky = cholesky_inverse(A)

# print("Inverse of original matrix A:")
# print(A_inverse)
# print("\nInverse of A using Cholesky decomposition:")
# print(A_cholesky)
# %%
#Comparing the FD Matrix and FD Conv

# %%
#Obtaining the Laplace using convolutions. 
import torch 
import torch.nn.functional as F

def fwd_laplace_stencil(X):    
    laplace_kernel = torch.tensor([[0., 1., 0.],
                                   [1., -4., 1.],
                                   [0, 1., 0.]])
    
    conv = F.conv2d(X, laplace_kernel.view(1,1,3,3),padding=(1,1))
    return conv

X_torch = torch.tensor(X, dtype=torch.float32)
fwd_laplace_conv = fwd_laplace_stencil(X_torch.view(1, 1, X_torch.shape[0], X_torch.shape[1]))[0,0]


# %%
#Comparing the Fourier based method. 
import numpy as np
from scipy.fft import fft2, ifft2

def laplace_operator_fft(field):
    """
    Computes the Laplace operator of a 2D field using Fast Fourier Transforms (FFT).
    
    Args:
        field (numpy.ndarray): A 2D array representing the field.
        
    Returns:
        numpy.ndarray: The Laplace operator of the input field.
    """
    # Compute the FFT of the input field
    fft_field = fft2(field)
    
    # Get the dimensions of the field
    m, n = field.shape
    
    # Create the frequency domain mesh
    col_mesh, row_mesh = np.meshgrid(np.fft.fftfreq(n, 2/n), np.fft.fftfreq(m, 2/m))
    
    # Compute the Laplacian operator in the frequency domain
    laplacian_kernel = -1 * (2 * np.pi) ** 2 * (col_mesh ** 2 + row_mesh ** 2)
    
    # Apply the Laplacian kernel to the FFT of the field
    fft_laplacian = fft_field * laplacian_kernel
    
    # Compute the inverse FFT to get the Laplacian in the spatial domain
    laplacian =  np.real(ifft2(fft_laplacian))
    
    return laplacian

fwd_laplace_fourier = laplace_operator_fft(X)

# %%

fig = plt.figure(figsize=(18, 5))

# mini = torch.min(X)
# maxi = torch.max(X)


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


ax = fig.add_subplot(1,3,1)
pcm =ax.imshow(fwd_laplace_soln, cmap=cm.coolwarm)#, vmin=mini, vmax=maxi)
ax.title.set_text('Matrix Stencil')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(1,3,2)
pcm =ax.imshow(fwd_laplace_conv, cmap=cm.coolwarm)#,  vmin=mini, vmax=maxi)
ax.title.set_text('Conv Stencil')
ax.set_xlabel('x')
# ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(1,3,3)
pcm =ax.imshow(fwd_laplace_fourier, cmap=cm.coolwarm)#,  vmin=mini, vmax=maxi)
ax.title.set_text('FFT')
ax.set_xlabel('x')
# ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

# %%
