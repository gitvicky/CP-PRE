#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------------------------
1D Burgers' Equation using the spectral method in Python 
---------------------------------------------------------------------------------------

Code inspired from from Steve Brunton's Lectures

Equation:
u_t + u*u_x =  nu*u_xx on [-1,1] x [-1,1]

Boundary Conditions: 
Periodic (implicit via FFT)

Initial Distribution :
u(x,y,t=0) = sin(alpha*pi*x) + np.cos(beta*np.pi*-x) + 1/np.cosh(gamma*np.pi*x) 
where alpha, beta, gamma all lie within the domain [-3,3]^3

Initial Velocity Condition :
u_t(x,y,t=0) = 0

"""
# %%
#Importing the necessary packages. 
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint

#RHS of the Burgers to be solved using odeint
def rhsBurgers(u, t, kappa, nu ):
    uhat = np.fft.fft(u)
    d_uhat = (1j)*kappa*uhat
    dd_uhat = -np.power(kappa, 2)*uhat
    d_u = np.fft.ifft(d_uhat)
    dd_u = np.fft.ifft(dd_uhat)
    du_dt = -u*d_u + nu*dd_u
    residual = (du_dt + u*d_u - nu*dd_u).mean()
                
    return du_dt.real


class Burgers_1D:

    def __init__(self,Nx, Nt, x_min, x_max, t_end, nu):
    
    # """
    # Initialize the Burgers 1D class
    # Args:
    #     Nx (int): Numerical discretiation in space. 
    #     Nt (int): Numerical discretiation in time. 
    #     x_min (float): The minimum value of the spatial domain.
    #     x_max (float): The maximum value of the spatial domain.
    #     t_lim (float): The maximum value of the time domain.
    #     nu (float): viscoity coefficient
    # """      
        self.Nx = Nx  
        self.Nt = Nt
        self.L = x_max
        self.t_end = t_end 
        self.nu = nu


        self.dx = self.L / self.Nx
        self.x = np.arange(0, self.L, self.dx) #Define the X Domain 

        self.dt = self.t_end / self.Nt

        #Define the discrete wavenumbers 
        self.kappa = 2*np.pi*np.fft.fftfreq(Nx, d=self.dx)

        #Initial Condition
        # self.InitializeU()

        #Simulate in Fourier Freq domain. 
        self.t= np.arange(0, self.t_end, self.dt)


    def InitializeU(self, alpha, beta, gamma):
        self.u0 = np.sin(alpha*np.pi*self.x) + np.cos(beta*np.pi*-self.x) + 1/np.cosh(gamma*np.pi*self.x) 

    def solve(self):
        u = odeint(rhsBurgers, self.u0, self.t, args=(self.kappa, self.nu))
        return u, self.x, self.dt

# %% 
# Example Usage 
if __name__ == "__main__":
    #Example Usage
    Nx = 1000 #Number of x-points
    Nt = 500 #Number of time instances 
    x_min = 0.0 #Min of X-range 
    x_max = 2.0 #Max of X-range 
    t_end = 1.25 #Time Maximum
    nu = 0.002

    alpha, beta, gamma = 1.0, 1.0, 1.0

    sim = Burgers_1D(Nx, Nt, x_min, x_max, t_end, nu) 
    sim.InitializeU(alpha, beta, gamma)
    u_sol = sim.solve()

# %%
