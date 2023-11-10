
"""

Modularised Spectral Navier Stokes Solver

Code optimised from https://levelup.gitconnected.com/create-your-own-navier-stokes-spectral-method-fluid-simulation-with-python-3f37405524f4

Equations: 
v_t + (v.nabla) v = nu * nabla^2 v + nabla P
div(v) = 0

"""
# %% 
#Setting up the configuration 
configuration = {"Spatial Resolution": 400, # Spatial resolution
				 "Start Time": 0.0,  # current time of the simulation
                 "End Time": 1.0, # time at which simulation ends
				 "Timestep": 0.0001, # timestep
				 "Viscosity": 0.001, # viscosity
				 "Domain Length": 1}
# %% 
#Importing the necessary packages 

import numpy as np
import matplotlib.pyplot as plt
from simvue import Run 
run = Run(mode='disabled')
run = run.init(folder="/NS_Spectral", tags=['NS', 'Spectral Solver', 'Periodic BC'], metadata=configuration)

# %% 
#Solver Functions 

def poisson_solve( rho, kSq_inv ):
	""" solve the Poisson equation, given source field rho """
	V_hat = -(np.fft.fftn( rho )) * kSq_inv
	V = np.real(np.fft.ifftn(V_hat))
	return V


def diffusion_solve( v, dt, nu, kSq ):
	""" solve the diffusion equation over a timestep dt, given viscosity nu """
	v_hat = (np.fft.fftn( v )) / (1.0+dt*nu*kSq)
	v = np.real(np.fft.ifftn(v_hat))
	return v


def grad(v, kx, ky):
	""" return gradient of v """
	v_hat = np.fft.fftn(v)
	dvx = np.real(np.fft.ifftn( 1j*kx * v_hat))
	dvy = np.real(np.fft.ifftn( 1j*ky * v_hat))
	return dvx, dvy


def div(vx, vy, kx, ky):
	""" return divergence of (vx,vy) """
	dvx_x = np.real(np.fft.ifftn( 1j*kx * np.fft.fftn(vx)))
	dvy_y = np.real(np.fft.ifftn( 1j*ky * np.fft.fftn(vy)))
	return dvx_x + dvy_y


def curl(vx, vy, kx, ky):
	""" return curl of (vx,vy) """
	dvx_y = np.real(np.fft.ifftn( 1j*ky * np.fft.fftn(vx)))
	dvy_x = np.real(np.fft.ifftn( 1j*kx * np.fft.fftn(vy)))
	return dvy_x - dvx_y


def apply_dealias(f, dealias):
	""" apply 2/3 rule dealias to field f """
	f_hat = dealias * np.fft.fftn(f)
	return np.real(np.fft.ifftn( f_hat ))

# %%

# Simulation parameters
N         = configuration['Spatial Resolution']  # Spatial resolution
t         = configuration["Start Time"]       # current time of the simulation
tEnd      = configuration["End Time"]      # time at which simulation ends
dt        = configuration['Timestep']   # timestep
tOut      = 0.01    # draw frequency
nu        = configuration['Viscosity']   # viscosity
plotRealTime = False # switch on for plotting as the simulation goes along

# Domain [0,1] x [0,1]
L = configuration['Domain Length']
xlin = np.linspace(0,L, num=N+1)  # Note: x=0 & x=1 are the same point!
xlin = xlin[0:N]                  # chop off periodic point
xx, yy = np.meshgrid(xlin, xlin)

# Intial Condition (vortex)
vx = -np.sin(2*np.pi*yy**2) 
vy =  np.cos(2*np.pi*xx**2) 

# Fourier Space Variables
klin = 2.0 * np.pi / L * np.arange(-N/2, N/2)
kmax = np.max(klin)
kx, ky = np.meshgrid(klin, klin)
kx = np.fft.ifftshift(kx)
ky = np.fft.ifftshift(ky)
kSq = kx**2 + ky**2
kSq_inv = 1.0 / kSq
kSq_inv[kSq==0] = 1

# dealias with the 2/3 rule
dealias = (np.abs(kx) < (2./3.)*kmax) & (np.abs(ky) < (2./3.)*kmax)

# number of timesteps
Nt = int(np.ceil(tEnd/dt))

# prep figure
fig = plt.figure(figsize=(4,4), dpi=80)
outputCount = 1

#Main Loop
for i in range(Nt):

    # Advection: rhs = -(v.grad)v
    dvx_x, dvx_y = grad(vx, kx, ky)
    dvy_x, dvy_y = grad(vy, kx, ky)
    
    rhs_x = -(vx * dvx_x + vy * dvx_y)
    rhs_y = -(vx * dvy_x + vy * dvy_y)
    
    rhs_x = apply_dealias(rhs_x, dealias)
    rhs_y = apply_dealias(rhs_y, dealias)

    vx += dt * rhs_x
    vy += dt * rhs_y
    
    # Poisson solve for pressure
    div_rhs = div(rhs_x, rhs_y, kx, ky)
    P = poisson_solve( div_rhs, kSq_inv )
    dPx, dPy = grad(P, kx, ky)
    
    # Correction (to eliminate divergence component of velocity)
    vx += - dt * dPx
    vy += - dt * dPy
    
    # Diffusion solve (implicit)
    vx = diffusion_solve( vx, dt, nu, kSq )
    vy = diffusion_solve( vy, dt, nu, kSq )
    
    # vorticity (for plotting)
    wz = curl(vx, vy, kx, ky)
    
    # update time
    t += dt
    print(t)
    
    # plot in real time
    plotThisTurn = False
    if t + dt > outputCount*tOut:
        plotThisTurn = True
    if (plotRealTime and plotThisTurn) or (i == Nt-1):
        
        plt.cla()
        plt.imshow(wz, cmap = 'RdBu')
        plt.clim(-20,20)
        ax = plt.gca()
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)	
        ax.set_aspect('equal')	
        plt.pause(0.001)
        outputCount += 1
        
        
# Save figure
# plt.savefig('navier-stokes-spectral.png',dpi=240)

