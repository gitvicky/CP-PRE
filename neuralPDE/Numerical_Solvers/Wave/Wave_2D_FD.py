# %%
import numpy as np 
from matplotlib import pyplot as plt 

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
    Nt = int(T/dt)

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


# # Problem parameters
# Lx = 1 # Length of the domain in x-direction
# Ly = 1  # Length of the domain in y-direction
# Nx = 128  # Number of grid points in x-direction
# Ny = 128  # Number of grid points in y-direction
# dt = 0.001  # Total simulation time
# Nt = 150 # Number of time steps
# c = 1.0   # Wave speed

# # Solve the wave equation
# u, dx, dy, dt = wave_equation_2d(Lx, Ly, Nx, Ny, dt, Nt, c)

# # Plot the solution at the final time step
# plt.imshow(u[-1], cmap='viridis', extent=[0, Lx, 0, Ly])
# plt.colorbar()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('2D Wave Equation - FD')
# plt.show()
# # %%