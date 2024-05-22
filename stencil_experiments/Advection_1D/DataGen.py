#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

---------------------------------------------------------------------------------------
Generating 1D Advection Dataset for PDE Surrogate Modelling 
---------------------------------------------------------------------------------------

"""
# %% 
#Importing the necessary packages 
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
from pyDOE import lhs 
# %% 
class Advection_1d:
    
    def __init__(self, xc, v, N=200, tmax=1.0):
        self.N = N # number of nodes
        self.tmax = tmax
        self.xmin = 0
        self.xmax = 2
        self.dt = 0.009 # timestep
        # self.v = 1 # velocity
        self.v = v
        # self.xc = 0.25  
        self.xc = xc
        self.amp = 200
        # self.amp = amp
        self.initializeDomain()
        self.initializeU()
        self.initializeParams()
        
        self.u_sol = []
        self.u_exact = []
        
    def initializeDomain(self):
        self.dx = (self.xmax - self.xmin)/self.N
        self.x = np.arange(self.xmin-self.dx, self.xmax+(2*self.dx), self.dx)
        
        
    def initializeU(self):
        u0 = np.exp(-self.amp*(self.x-self.xc)**2)
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
            
            # uexact = np.exp(-200*(self.x-self.xc-self.v*tc)**2)
            # self.u_exact.append(uexact)
            
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
            
        # return self.x, np.linspace(0, self.tmax, self.nsteps), np.asarray(self.u_sol), np.asarray(self.u_exact)   
        return self.u_sol

# %%
#Obtaining the exact and FD solution of the 1D Advection Equation. 
x_discretisation, t_end = 200, 0.5
n_sims = 100

lb = np.asarray([0.1, 0.1]) #pos, velocity
ub = np.asarray([1.0, 1.0])

params = lb + (ub - lb) * lhs(2, n_sims)

# %% 
u_sol = []
for ii in tqdm(range(n_sims)):

    solver = Advection_1d(xc=params[ii, 0], v=params[ii,1], N=x_discretisation, tmax=t_end)
    u_sol.append(solver.solve())

u_sol = np.asarray(u_sol)
u_sol = u_sol[:, :, 1:-2]
dt = solver.dt
dx = solver.dx
x = solver.x[1:-2]
t = np.linspace(0, solver.tmax, solver.nsteps)
v = params[:,1]
# %%

