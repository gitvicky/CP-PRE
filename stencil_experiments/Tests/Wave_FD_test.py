# %% 
import numpy as np
import matplotlib.pyplot as plt

def wave_equation_2d(Lx, Ly, Nx, Ny, T, Nt, c):
    """
    Solve the 2D wave equation using the finite difference method.

    Args:
        Lx (float): Length of the domain in x-direction.
        Ly (float): Length of the domain in y-direction.
        Nx (int): Number of grid points in x-direction.
        Ny (int): Number of grid points in y-direction.
        T (float): Total simulation time.
        Nt (int): Number of time steps.
        c (float): Wave speed.

    Returns:
        u_all (numpy.ndarray): Solution at all time steps.
    """
    # Spatial and temporal step sizes
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dt = T / Nt

    # Initialize the solution arrays
    u = np.zeros((Nx, Ny))      # Solution at current time step
    u_prev = np.zeros((Nx, Ny)) # Solution at previous time step
    u_next = np.zeros((Nx, Ny)) # Solution at next time step
    u_all = np.zeros((Nt + 1, Nx, Ny)) # Solution at all time steps

    # Set initial conditions
    def initial_condition(xx, yy):
        return np.exp(-50*((xx-0.5)**2 + (yy-0.5)**2))

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u[:] = initial_condition(X, Y)
    u_all[0] = u

    # Time-stepping loop
    for n in range(1, Nt + 1):
        # Update the solution at the next time step
        u_next[1:-1, 1:-1] = (
            2 * u[1:-1, 1:-1] - u_prev[1:-1, 1:-1]
            + (c * dt / dx)**2 * (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1])
            + (c * dt / dy)**2 * (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2])
        )

        # Update the solution at the current and previous time steps
        u_prev[:] = u[:]
        u[:] = u_next[:]

        # Apply Dirichlet boundary conditions
        u[0, :] = 0
        u[-1, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0

        # Store the solution at the current time step
        u_all[n] = u

    return u_all, dx, dy, dt

def plot_solution(u_all, Lx, Ly, Nt):
    """
    Plot the solution at different time steps.

    Args:
        u_all (numpy.ndarray): Solution at all time steps.
        Lx (float): Length of the domain in x-direction.
        Ly (float): Length of the domain in y-direction.
        Nt (int): Number of time steps.
    """
    num_plots = 4
    time_indices = np.linspace(0, Nt, num_plots, dtype=int)

    fig, axes = plt.subplots(1, num_plots, figsize=(12, 3))
    for i, ax in enumerate(axes):
        ax.imshow(u_all[time_indices[i]], cmap='jet', origin='lower', extent=[0, Lx, 0, Ly])
        ax.set_title(f'Time step: {time_indices[i]}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout()
    plt.show()

# Problem parameters
Lx = 1.0  # Length of the domain in x-direction
Ly = 1.0  # Length of the domain in y-direction
Nx = 128  # Number of grid points in x-direction
Ny = 128  # Number of grid points in y-direction
T = 0.5  # Total simulation time
Nt = 100 # Number of time steps
c = 1.0   # Wave speed

# Solve the wave equation
u_all, dx, dy, dt = wave_equation_2d(Lx, Ly, Nx, Ny, T, Nt, c)

# Plot the solution at different time steps
plot_solution(u_all, Lx, Ly, Nt)
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


CS_stencil_2nd = torch.tensor([[0, -1, 0],
                           [0, 0, 0],
                           [0, 1, 0]], dtype=torch.float32)
                           

u_tensor = torch.tensor(u_all, dtype=torch.float32)


def conv_deriv_2d(f, stencil):
    return F.conv2d(f.unsqueeze(0).unsqueeze(0), stencil.unsqueeze(0).unsqueeze(0), padding=stencil.shape[0]//2).squeeze()

u_xx_conv_2d = conv_deriv_2d(u_tensor[-10], three_p_stencil)
u_yy_conv_2d = conv_deriv_2d(u_tensor[-10], three_p_stencil.T)
u_xx_yy_conv_2d  = conv_deriv_2d(u_tensor[-10], laplacian_stencil_2nd)
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
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#There is some issue with the padding implementation here for the spatial domain
def conv_deriv_3d(f, stencil):
    return F.conv3d(f.unsqueeze(0).unsqueeze(0), stencil.unsqueeze(0).unsqueeze(0), padding=(stencil.shape[0]//2, stencil.shape[1]//2, stencil.shape[2]//2)).squeeze()

kernel = kernel_3d(laplacian_stencil_2nd, axis=0)
convolution = F.conv3d(u_tensor.view(1, 1, Nt+1, Nx, Ny), kernel.view(1, 1, kernel.shape[0], kernel.shape[1], kernel.shape[2]), padding=(1,1,1))[0,0]

u_xx_yy_conv_3d = conv_deriv_3d(u_tensor, kernel_3d(laplacian_stencil_2nd, axis=0))[:, 1:-1,1:-1]
u_tt_conv_3d = conv_deriv_3d(u_tensor, kernel_3d(three_p_stencil, axis=2))[:, 1:-1,1:-1]

# %%

from matplotlib import pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
fig = plt.figure(figsize=(16, 4))
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
pcm =ax.imshow(u_xx_yy_conv_3d[idx], cmap='jet', origin='lower', extent=[0, Lx, 0, Ly])#, vmin=mini, vmax=maxi)
ax.title.set_text(r'$(u_{xx} + u_{yy})$')
ax.set_xlabel('x')
ax.set_ylabel('CK as FDS')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(1,3,2)
pcm =ax.imshow(u_tt_conv_3d[idx], cmap='jet', origin='lower', extent=[0, Lx, 0, Ly])#,  vmin=mini, vmax=maxi)
ax.title.set_text(r'$u_t$')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(1,3,3)
pcm =ax.imshow(u_tt_conv_3d[idx] - (c*dt/dx)**2*u_xx_yy_conv_3d[idx], cmap='jet', origin='lower', extent=[0, Lx, 0, Ly])#,  vmin=mini, vmax=maxi)
ax.title.set_text(r'$Residual$')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

# %%
