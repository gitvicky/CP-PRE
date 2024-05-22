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
    
    def __init__(self, N, tmax):
        self.N = N # number of nodes
        self.tmax = tmax
        self.xmin = 0
        self.xmax = 2
        self.dt = 0.009 # timestep
        self.v = 1 # velocity
        self.xc = 0.25
        self.initializeDomain()
        self.initializeU()
        self.initializeParams()
        
        self.u_sol = []
        self.u_exact = []
        
    def initializeDomain(self):
        self.dx = (self.xmax - self.xmin)/self.N
        self.x = np.arange(self.xmin-self.dx, self.xmax+(2*self.dx), self.dx)
        
        
    def initializeU(self):
        u0 = np.exp(-200*(self.x-self.xc)**2)
        self.u = u0.copy()
        self.unp1 = u0.copy()
        
        
    def initializeParams(self):
        self.nsteps = round(self.tmax/self.dt)
        self.alpha = self.v*self.dt/(2*self.dx)
        
        
    def solve(self):
        tc = 0
        
        for i in range(self.nsteps):
            # plt.clf()
            
            # The Lax-Wendroff scheme, Eq. (18.20)
            for j in range(self.N+2):
                self.unp1[j] = self.u[j] + (self.v**2*self.dt**2/(2*self.dx**2))*(self.u[j+1]-2*self.u[j]+self.u[j-1]) \
                - self.alpha*(self.u[j+1]-self.u[j-1])
                
            self.u = self.unp1.copy()
            
            # Periodic boundary conditions
            self.u[0] = self.u[self.N+1]
            self.u[self.N+2] = self.u[1]
            
            uexact = np.exp(-200*(self.x-self.xc-self.v*tc)**2)
            self.u_exact.append(uexact)
            
            # plt.plot(self.x, uexact, 'r', label="Exact solution")
            # plt.plot(self.x, self.u, 'bo-', label="Lax-Wendroff")
            # plt.axis((self.xmin-0.12, self.xmax+0.12, -0.2, 1.4))
            # plt.grid(True)
            # plt.xlabel("Distance (x)")
            # plt.ylabel("u")
            # plt.legend(loc=1, fontsize=12)
            # plt.suptitle("Time = %1.3f" % (tc+self.dt))
            # plt.pause(0.01)
            tc += self.dt
            
            self.u_sol.append(self.u)
            
        return self.x, np.linspace(0, self.tmax, self.nsteps), np.asarray(self.u_sol), np.asarray(self.u_exact)


    
if __name__ == "__main__":
    sim = Advection_1d(100, 0.5) # Discreisation and T_end
    x, t, u_sol, u_exact = sim.solve()
    v = 1 
    

# %%
