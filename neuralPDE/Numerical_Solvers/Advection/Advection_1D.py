"""
This code solves the advection equation using the Lax-Friedrichs method

    U_t + v U_x = 0

over the spatial domain of 0 <= x <= 1 that is discretized 
into 103 nodes, with dx=0.01 for an initial profile of a Gaussian curve, 
defined by 

    U(x,t) = exp(-200*(x-xc-v*t).^2)

where xc=0.25 is the center of the curve at t=0.

The periodic boundary conditions are applied either end of the domain.
The velocity is v=1. The solution is iterated until t=1.5 seconds.

Lax-Friedrichs - https://en.wikipedia.org/wiki/Lax%E2%80%93Friedrichs_method 
"""

# %% 
import numpy as np
import matplotlib.pyplot as plt


class Advection_1d:
    
    def __init__(self, Nx, Nt, x_min, x_max, t_end):
       
        """
        Initialize the Advection_1d class.

        Args:
            Nx (int): Number of x-points.
            Nt (int): Number of time instances.
            x_min (float): Minimum value of x.
            x_max (float): Maximum value of x.
            t_end (float): Time length.
            v (float): Advection velocity.
            xc (float): Center of the Gaussian curve at t=0.
        """

        self.Nx = Nx
        self.x_min = x_min
        self.x_max = x_max
        self.dx = (x_max - x_min)/self.Nx
        self.x = np.arange(x_min, x_max, self.dx)

        self.t_length = Nt
        self.dt = (t_end)/self.t_length
        self.t = np.arange(0, t_end, self.dt)
        self.tmax = t_end

        self.initializeDomain()
        
        
    def initializeDomain(self):
        """
        Initialize the spatial domain.
        """
        self.dx = (self.x_max - self.x_min)/self.Nx
        self.x = np.arange(self.x_min-self.dx, self.x_max+(2*self.dx), self.dx)
        
        
    def initializeU(self, xc, amp):
        """
        Initialize the solution array U and the next time step array unp1.
        """
        u0 = np.exp(-amp*(self.x-xc)**2)
        self.u = u0.copy()
        self.unp1 = u0.copy()
        
        
    def initializeParams(self):
        """
        Initialize the simulation parameters and check the CFL condition.
        """
        self.nsteps = round(self.tmax/self.dt)
        self.alpha = self.v*self.dt/(2*self.dx)
        
        # Print the Courant number
        courant_number = self.v * self.dt / self.dx
        # print(f"Courant number: {courant_number}")
        
        # Assert that the CFL condition is not violated
        assert courant_number <= 1, "CFL condition violated"
        
    def solve(self, xc, amp, v=1):

        self.v = v # velocity
        self.xc = xc #position of the gaussian
        self.amp = amp #amplitude of the gaussian

        self.initializeU(self.xc, self.amp)
        self.initializeParams()
        
        self.u_sol = []
        self.u_exact = []

        """
        Solve the advection equation using the Lax-Friedrichs method.

        Returns:
            tuple: A tuple containing the following elements:
                - self.x (np.ndarray): x-coordinates.
                - np.linspace(0, self.tmax, self.nsteps) (np.ndarray): Time steps.
                - np.asarray(self.u_sol) (np.ndarray): Solution at each time step.
                - np.asarray(self.u_exact) (np.ndarray): Exact solution at each time step (Analytical).
        """
        tc = 0
        
        for i in range(self.nsteps):
            
            # The Lax-Wendroff scheme
            for j in range(self.Nx+2):
                self.unp1[j] = self.u[j] + (self.v**2*self.dt**2/(2*self.dx**2))*(self.u[j+1]-2*self.u[j]+self.u[j-1]) \
                - self.alpha*(self.u[j+1]-self.u[j-1])
                
            self.u = self.unp1.copy()
            
            # Periodic boundary conditions
            self.u[0] = self.u[self.Nx+1]
            self.u[self.Nx+2] = self.u[1]
            
            uexact = np.exp(-200*(self.x-self.xc-self.v*tc)**2)
            self.u_exact.append(uexact)
            
            tc += self.dt
            
            self.u_sol.append(self.u)
            
        return self.x, np.linspace(0, self.tmax, self.nsteps), np.asarray(self.u_sol), np.asarray(self.u_exact)


# if __name__ == "__main__":
#     #Example Usage 
#     Nx = 100 #Number of x-points
#     Nt = 50 #Number of time instances 
#     x_min, x_max = 0.0, 2.0 #X min and max
#     t_end = 0.5 #time length
#     v = 1 #Advection velocity 
#     xc = 0.25 #Centre of Gaussian 
#     amp = 200 #Amplitude

#     sim = Advection_1d(Nx, Nt, x_min, x_max, t_end) 
#     x, t, u_sol, u_exact = sim.solve(xc, amp, v)
#     v = 1 
#     plt.plot(x, u_sol.T)
#     plt.xlabel('x')
#     plt.ylabel('u: in-time')
# %%