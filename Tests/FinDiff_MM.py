
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
import scipy
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
# Generating the wave data through simulation:

import sys 
sys.path.append('/Users/Vicky/Documents/UKAEA/Code/Uncertainty_Quantification/PDE_Residuals/Neural_PDE')

Nx = 33 # Mesh Discretesiation 
Nt = 100 #Max time
x_min = -1.0 # Minimum value of x
x_max = 1.0 # maximum value of x
y_min = -1.0 # Minimum value of y 
y_max = 1.0 # Minimum value of y
tend = 1
Lambda = 40 #Gaussian Amplitude
aa = 0.0 #X-pos
bb = 0.0 #Y-pos
c = 1.0 #Wave Speed <=1.0

#Initialising the Solver
from Neural_PDE.Numerical_Solvers.Wave import Wave_2D_Spectral
solver = Wave_2D_Spectral.Wave_2D(Nx, Nt, x_min, x_max, tend, c, Lambda, aa , bb)

#Solving and obtaining the solution. 
x, y, t, u_sol = solver.solve() #solution shape -> t, x, y

# %% 
# #Example Usage 

dx = x[1] - x[0]
dt = t[1] - t[0]

laplace_stencil = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]])  # 2D Laplacian stencil-

# stencil_center = (1, 1)  # Center of the stencil
nx, ny = Nx, Nx

# %% 
fwd_laplace = W = finite_difference_matrix_2d(nx, ny, stencil, stencil_center) #fwd_laplace
inv_laplace = np.linalg.inv(W)

# %% 
#In 3D for a field that is characterised as t, x, y

import numpy as np

def finite_difference_matrix_3d(nt, nx, ny, stencil, stencil_center):
    """
    Constructs the finite difference matrix for a given 3D stencil and grid sizes.

    Args:
        nt (int): The number of grid points in the t-direction.
        nx (int): The number of grid points in the x-direction.
        ny (int): The number of grid points in the y-direction.
        stencil (np.ndarray): A 3D array representing the finite difference stencil.
        stencil_center (tuple): The center of the stencil, given as (t_center, x_center, y_center).

    Returns:
        np.ndarray: The finite difference matrix.
    """
    kt, kx, ky = stencil.shape  # Order of the stencil in t, x, and y directions
    n = nt * nx * ny  # Total number of grid points
    A = np.zeros((n, n))
    t_center, x_center, y_center = stencil_center  # Center of the stencil

    # Fill the matrix rows
    for k in range(nt):
        for i in range(nx):
            for j in range(ny):
                row_idx = (k * nx + i) * ny + j  # Linear index for the current grid point

                for kt_idx in range(kt):
                    for kx_idx in range(kx):
                        for ky_idx in range(ky):
                            # Compute the neighbor indices
                            kt = k + kt_idx - t_center
                            ix = i + kx_idx - x_center
                            jy = j + ky_idx - y_center

                            # Skip out-of-bounds neighbors
                            if kt < 0 or kt >= nt or ix < 0 or ix >= nx or jy < 0 or jy >= ny:
                                continue

                            col_idx = (kt * nx + ix) * ny + jy  # Linear index for the neighbor
                            A[row_idx, col_idx] = stencil[kt_idx, kx_idx, ky_idx]

    return A

# %% 
#Example Usage 

# Define the grid sizes and stencil
nt, nx, ny = 32, 64, 64
stencil = np.array([
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
])
stencil_center = (1, 1, 1)

# Create the finite difference matrix
fwd_laplace_3d = finite_difference_matrix_3d(nt, nx, ny, stencil, stencil_center)

# Apply the matrix to a 3D tensor
tensor = np.random.rand(nt, nx, ny)
fwd_laplace_soln_3d = np.matmul(fwd_laplace_3d, tensor.reshape(-1)).reshape(nt, nx, ny)
# %%
