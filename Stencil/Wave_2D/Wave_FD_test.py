# %% 

import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx = 1.0  # Length of the domain in x-direction
Ly = 1.0  # Length of the domain in y-direction
Nx = 128  # Number of grid points in x-direction
Ny = 128  # Number of grid points in y-direction
dt = 0.001  # Time step
Nt = 500  # Number of time steps
c = 1.0  # Wave speed

# Grid spacing
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

# Grid points
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initial condition (2D Gaussian)
sigma = 0.1
x0 = Lx / 2
y0 = Ly / 2
u0 = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

# Initialize solution arrays
u = np.zeros((Nt, Ny, Nx))
u[0] = u0
u[1] = u0

# Finite difference scheme
for n in range(1, Nt - 1):
    for i in range(1, Ny - 1):
        for j in range(1, Nx - 1):
            u[n+1, i, j] = 2 * u[n, i, j] - u[n-1, i, j] + (c * dt / dx)**2 * (u[n, i+1, j] - 2 * u[n, i, j] + u[n, i-1, j]) + (c * dt / dy)**2 * (u[n, i, j+1] - 2 * u[n, i, j] + u[n, i, j-1])

    # Dirichlet boundary conditions
    u[n+1, :, 0] = 0
    u[n+1, :, -1] = 0
    u[n+1, 0, :] = 0
    u[n+1, -1, :] = 0

# Plot the solution at the final time step
plt.figure()
plt.contourf(X, Y, u[-1], cmap='plasma')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Wave Equation Solution')
plt.show()
# %%

import torch
import torch.nn.functional as F 

alpha = 1/dx**2
beta = 1/dt**2

three_p_stencil = torch.tensor([[0, 1, 0],
                           [0, -2, 0],
                           [0, 1, 0]], dtype=torch.float32)

laplacian_stencil_2nd = torch.tensor([[0., 1., 0.],
                       [1., -4., 1.],
                       [0., 1., 0.]])

laplacian_stencil_4th = torch.tensor([[0, 0, -1/12, 0, 0],
                                       [0, 0, 4/3, 0, 0],
                                       [-1/12, 4/3, -5/2, 4/3, -1/12],
                                       [0, 0, 4/3, 0, 0],
                                       [0, 0, -1/12, 0, 0]]
                                       )

laplacian_stencil_6th = torch.tensor([[0, 0, 0, 1/90, 0, 0, 0],
                                      [0, 0, 0, -3/20, 0, 0, 0],
                                      [0, 0, 0, 3/2, 0, 0, 0],
                                      [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90],
                                      [0, 0, 0, 3/2, 0, 0, 0],
                                      [0, 0, 0, -3/20, 0, 0, 0],
                                      [0, 0, 0, 1/90, 0, 0, 0]], dtype=torch.float32)

laplacian_stencil_8th = torch.tensor([
    [-9,    0,     0,     0,     0,     0,     0,     0,    -9],
    [0,   128,     0,     0,     0,     0,     0,   128,     0],
    [0,     0, -1008,     0,     0,     0, -1008,     0,     0],
    [0,     0,     0,  8064,     0,  8064,     0,     0,     0],
    [0,     0,     0,     0, -14350,     0,     0,     0,     0],
    [0,     0,     0,  8064,     0,  8064,     0,     0,     0],
    [0,     0, -1008,     0,     0,     0, -1008,     0,     0],
    [0,   128,     0,     0,     0,     0,     0,   128,     0],
    [-9,    0,     0,     0,     0,     0,     0,     0,    -9]
    ],dtype=torch.float32)/5040

laplacian_stencil_biharmonic = torch.tensor([[0,0,1,0,0],
                                           [0,2,-8,2,0],
                                           [1,-8,1,-8,1],
                                           [0,2,-8,2,0],
                                           [0,0,1,0,0]], dtype=torch.float32)

CS_stencil_2nd = torch.tensor([[0, -1, 0],
                           [0, 0, 0],
                           [0, 1, 0]], dtype=torch.float32)
                           

u_tensor = torch.tensor(u, dtype=torch.float32)


def conv_deriv_2d(f, stencil):
    return F.conv2d(f.unsqueeze(0).unsqueeze(0), stencil.unsqueeze(0).unsqueeze(0), padding=stencil.shape[0]//2).squeeze()

# u_xx_conv_2d = conv_deriv_2d(u_tensor[-10], three_p_stencil)
# u_yy_conv_2d = conv_deriv_2d(u_tensor[-10], three_p_stencil.T)
# u_xx_yy_conv_2d  = conv_deriv_2d(u_tensor[-10], laplacian_stencil_4th)
# %%
def kernel_3d(stencil, axis):
    kernel_size = stencil.shape[0]
    kernel = torch.zeros(kernel_size, kernel_size, kernel_size)
    if axis == 0:
        kernel[1,:,:] = stencil
    elif axis ==1:
            kernel[:,1,:] = stencil
    elif axis ==2:
            kernel[:,:,1] = stencil
    return kernel

# %%%
def conv_deriv_3d(f, stencil):
    return F.conv3d(f.unsqueeze(0).unsqueeze(0), stencil.unsqueeze(0).unsqueeze(0), padding=(stencil.shape[0]//2, stencil.shape[1]//2, stencil.shape[2]//2)).squeeze()

u_xx_yy_conv_3d = conv_deriv_3d(u_tensor, kernel_3d(laplacian_stencil_4th, axis=1))
u_tt_conv_3d = conv_deriv_3d(u_tensor, kernel_3d(laplacian_stencil_4th, axis=2))

# %%

from matplotlib import pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
fig = plt.figure(figsize=(12, 4))
idx = -10

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
pcm =ax.imshow(u_xx_yy_conv_3d[idx], cmap=cm.coolwarm)#, vmin=mini, vmax=maxi)
ax.title.set_text(r'$(u_{xx} + u_{yy})$')
ax.set_xlabel('x')
ax.set_ylabel('CK as FDS')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(1,3,2)
pcm =ax.imshow(u_tt_conv_3d[idx], cmap=cm.coolwarm)#,  vmin=mini, vmax=maxi)
ax.title.set_text(r'$u_t$')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(1,3,3)
pcm =ax.imshow(u_tt_conv_3d[idx] - (c*dt/dx)**2*u_xx_yy_conv_3d[idx], cmap=cm.coolwarm)#,  vmin=mini, vmax=maxi)
ax.title.set_text(r'$Residual')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

# %%
