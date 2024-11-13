#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D wave equation via FFT 

u_tt = c^2 * (u_xx + u_yy)

on [-1, 1]x[-1, 1], t > 0 and Dirichlet BC u=0

"""
# %%
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm  
from tqdm import tqdm 
 
class Wave_2D:
    def __init__(self, Nx, x_min, x_max, t_end, c):

        self.N =  Nx - 1 
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = x_min
        self.y_max = x_max
        self.dx = (x_max - x_min)/self.N
        self.dy = self.dx
        self.tend = t_end
        self.dt = 6/self.N**2
        self.t = np.arange(0,self.tend,self.dt)

        self.c = c # Wave Speed <=1.0
        assert self.c <= 1, "Unrealistic Wave Speed"

    def initialise(self, Lambda, aa, bb):
        """
        Initialize the grid, time step, and initial conditions.
        """
        k = np.arange(self.N + 1)#Wavenumber indexes
        self.x = np.cos(k*np.pi/self.N) #Creating the x and y discretisations
        self.y = self.x.copy()
        self.xx, self.yy = np.meshgrid(self.x, self.y)#Creating the 2D meshgrids
        
        
        #Initial Conditions 
        self.vv = np.exp(-Lambda*((self.xx-aa)**2 + (self.yy-bb)**2))
        self.vvold = self.vv.copy()
        
        self.nstep = int(self.tend / self.dt) + 1

    def solve(self, Lambda, aa, bb):

        self.initialise(Lambda, aa, bb)

        """
        Solve the 2D wave equation using the spectral method.

        Returns:
            tuple: A tuple containing the following elements:
                - xxx (np.ndarray): x-coordinates of the solution. 
                - yyy (np.ndarray): y-coordinates of the solution. 
                - self.t (np.ndarray): Time steps of the solution.  
                - u_sol (np.ndarray): Solution at each time step.
        """

        u_list = []
        tc = 0 
        while tc < self.nstep:

            #With Grid interpolation 
            xxx = np.arange(self.x_min, self.x_max+self.dx, self.dx)
            yyy = np.arange(self.y_min, self.y_max+self.dy, self.dy)
            vvv = interpolate.interp2d(self.x, self.y, self.vv, kind='cubic')
            Z = vvv(xxx, yyy)

            #Need to fix this to be in line with the latest scipy versions
            # vvv = interpolate.RegularGridInterpolator((self.x, self.y), self.vv, method='cubic')
            # Z = vvv((xxx, yyy))
            
            # #Without any interpolation
            # xxx = np.linspace(solver.x_min, solver.x_max, solver.N)
            # yyy = np.linspace(solver.x_min, solver.x_max, solver.N)
            # Z = self.vv
                
            uxx = np.zeros((self.N+1, self.N+1))
            uyy = np.zeros((self.N+1, self.N+1))
            ii = np.arange(1, self.N)
            
            for i in range(1, self.N):
                v = self.vv[i,:]
                V = np.hstack((v, np.flipud(v[ii])))
                U = np.fft.fft(V)
                U = U.real
                
                r1 = np.arange(self.N)
                r2 = 1j*np.hstack((r1, 0, -r1[:0:-1]))*U
                W1 = np.fft.ifft(r2)
                W1 = W1.real
                s1 = np.arange(self.N+1)
                s2 = np.hstack((s1, -s1[self.N-1:0:-1]))
                s3 = -s2**2*U
                W2 = np.fft.ifft(s3)
                W2 = W2.real
                
                uxx[i,ii] = W2[ii]/(1-self.x[ii]**2) - self.x[ii]*W1[ii]/(1-self.x[ii]**2)**(3/2)
                
            for j in range(1, self.N):
                v = self.vv[:,j]
                V = np.hstack((v, np.flipud(v[ii])))
                U = np.fft.fft(V)
                U = U.real
            
                r1 = np.arange(self.N)
                r2 = 1j*np.hstack((r1, 0, -r1[:0:-1]))*U
                W1 = np.fft.ifft(r2)
                W1 = W1.real
                s1 = np.arange(self.N+1)
                s2 = np.hstack((s1, -s1[self.N-1:0:-1]))
                s3 = -s2**2*U
                W2 = np.fft.ifft(s3)
                W2 = W2.real
                
                uyy[ii,j] = W2[ii]/(1-self.y[ii]**2) - self.y[ii]*W1[ii]/(1-self.y[ii]**2)**(3/2)
                
            self.vvnew = 2*self.vv - self.vvold + self.c**2*self.dt**2*(uxx+uyy)
            self.vvold = self.vv.copy()
            self.vv = self.vvnew.copy()
            tc += 1
            
            u_list.append(Z)
            
        u_sol = np.asarray(u_list)
        return xxx, yyy, self.t, u_sol

# %%
# #Example of Usage
# Nx = 32 # Mesh Discretesiation 
# x_min = -1.0 # Minimum value of x
# x_max = 1.0 # maximum value of x
# y_min = -1.0 # Minimum value of y 
# y_max = 1.0 # Minimum value of y
# tend = 1
# Lambda = 20
# aa = 0.25
# bb = 0.25
# c = 1.0 # Wave Speed <=1.0

# #Initialising the Solver
# solver = Wave_2D(Nx, x_min, x_max, tend, c)

# #Solving and obtaining the solution. 
# xx, yy, t, u_sol = solver.solve(Lambda, aa , bb) #solution shape -> t, x, y
# # %%

# # Plot the solution at the final time step
# plt.imshow(u_sol[-1], cmap='viridis', extent=[x_min, x_max, y_min, y_max])
# plt.colorbar()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('2D Wave Equation - Spectral FFT Solver')
# plt.show()
# # %%
