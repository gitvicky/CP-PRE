#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D wave equation via FFT 

u_tt = c^2 * (u_xx + u_yy)

on [-1, 1]x[-1, 1], t > 0 and Dirichlet BC u=0

Source : http://people.bu.edu/andasari/courses/numericalpython/python.html
"""
# %%
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm  
from tqdm import tqdm 
 

class Wave_2D:
    def __init__(self, Lambda, a, b):

        self.N = 30 # Mesh Discretesiation 
        self.x0 = -1.0 # Minimum value of x
        self.xf = 1.0 # maximum value of x
        self.y0 = -1.0 # Minimum value of y 
        self.yf = 1.0 # Minimum value of y
        self.tend = 1
        self.Lambda = Lambda
        self.a = a 
        self.b = b 
        self.c = 1.0 # Wave Speed <=1.0

        self.intialise()

    def intialise(self):
        k = np.arange(self.N + 1)
        self.x = np.cos(k*np.pi/self.N) #Creating the x and y discretisations
        self.y = self.x.copy()
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        
        dt = 6/self.N**2 # dont know why this is taken as dt 
        plotgap = round((1/3)/dt) 
        self.dt = (1/3)/plotgap
        
        #Initial Conditions 
        self.vv = np.exp(-self.Lambda*((self.xx-self.a)**2 + (self.yy-self.b)**2))
        self.vvold = self.vv.copy()
        
        
        self.nstep = round(3*plotgap+1) * self.tend
        self.t = np.arange(0,self.tend+self.dt,self.dt)

    def solve(self):
        u_list = []
        tc = 0 
        while tc < self.nstep:

            xxx = np.arange(self.x0, self.xf+1/16, 1/16)
            yyy = np.arange(self.y0, self.yf+1/16, 1/16)
            vvv = interpolate.interp2d(self.x, self.y, self.vv, kind='cubic')
            Z = vvv(xxx, yyy)
                
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
# Lambda = 20 #
# a = 0.25 #x-position of initial gaussian
# b = 0.25 #y-position of initial gaussian 


# solver = Wave_2D(Lambda, a , b)
# xx, yy, t, u_sol = solver.solve() #solution shape -> t, x, y
# %%
