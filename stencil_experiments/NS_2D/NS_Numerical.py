"""

Modularised Spectral Navier Stokes Solver

Code optimised from https://levelup.gitconnected.com/create-your-own-navier-stokes-spectral-method-fluid-simulation-with-python-3f37405524f4

Equations: 
v_t + (v.nabla) v = nu * nabla^2 v + nabla P
div(v) = 0

# Domain [0,1] x [0,1]


"""

# %% 
#Importing the necessary packages 

import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger("test")
logger.setLevel(level=logging.DEBUG)

logFileFormatter = logging.Formatter(
    fmt=f"%(levelname)s %(asctime)s (%(relativeCreated)d) \t %(pathname)s F%(funcName)s L%(lineno)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
fileHandler = logging.FileHandler(filename='test.log')
fileHandler.setFormatter(logFileFormatter)
fileHandler.setLevel(level=logging.INFO)

logger.addHandler(fileHandler)
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
class Navier_Stokes_2d:
	def __init__(self, N, t, tEnd, dt, nu, L, a, b):
		self.N = N
		self.t = t
		self.tEnd = tEnd
		self.dt = dt 
		self.nu = nu 
		self.L = L 

		self.xlin = np.linspace(0, L, num=N+1)
		self.xlin = self.xlin[0:N]
		self.xx, self.yy = np.meshgrid(self.xlin, self.xlin)

		self.vx = -np.sin(2*a*np.pi*self.yy)
		self.vy = np.sin(2*b*np.pi*self.xx*2)

		klin = 2*np.pi / self.L * np.arange(-self.N/2, self.N/2)
		kmax = np.max(klin)
		self.kx, self.ky = np.meshgrid(klin, klin)
		self.kx = np.fft.ifftshift(self.kx)
		self.ky = np.fft.ifftshift(self.ky)
		self.kSq = self.kx**2 + self.ky**2 
		self.kSq_inv = 1.0 / self.kSq
		self.kSq_inv[self.kSq==0] = 1 

		self.dealias = (np.abs(self.kx) < (2./3.)*kmax) & (np.abs(self.ky) < (2./3.)*kmax)

		self.Nt = int(np.ceil(self.tEnd/dt))


	def solve(self):
		u_list = []
		v_list = []
		p_list = []
		w_list = []
		error = 0 
		for ii in range(self.Nt):
			
			# Advection: rhs = -(v.grad)v
			dvx_x, dvx_y = grad(self.vx, self.kx, self.ky)
			dvy_x, dvy_y = grad(self.vy, self.kx, self.ky)

			rhs_x = -(self.vx * dvx_x + self.vy * dvx_y)
			rhs_y = -(self.vx * dvy_x + self.vy * dvy_y)

			rhs_x = apply_dealias(rhs_x, self.dealias)
			rhs_y = apply_dealias(rhs_y, self.dealias)

			self.vx += self.dt * rhs_x
			self.vy += self.dt * rhs_y

			#Poisson solve for Pressure
			div_rhs = div(rhs_x, rhs_y, self.kx, self.ky)
			P = poisson_solve(div_rhs, self.kSq_inv)
			dPx, dPy = grad(P, self.kx, self.ky)

			# Correction (to eliminate divergence component of velocity)
			self.vx += -self.dt * dPx
			self.vy += -self.dt * dPy

			#Diffusion solved Implcitly 
			self.vx = diffusion_solve(self.vx, self.dt , self.nu, self.kSq)
			self.vy = diffusion_solve(self.vy, self.dt, self.nu, self.kSq)

			#Vorticity
			wz = curl(self.vx, self.vy, self.kx, self.ky)

			#Continuity Residual
			cont = np.sum(dvx_x + dvy_y)

			#update time
			self.t += self.dt
			print("Iteration: {}, Time: {}, Residuals {}".format(ii, self.t , cont))

			if cont> 1: 
				logger.error('Numerical instability occured ! ')
				error =1 
				# break
			else:
				logger.info("Iteration: {}, Time: {}, Residuals {}".format(ii, self.t , cont))

			
			u_list.append(self.vx)
			v_list.append(self.vy)
			p_list.append(P)
			w_list.append(wz)

		return np.asarray(u_list), np.asarray(v_list), np.asarray(p_list), np.asarray(w_list), self.xlin, self.dt, error

# %% 

# solver= Navier_Stokes_2d(400, 0.0, 1.0, 0.0001, 0.001, 1) # N, tStart, tEnd, dt, nu, L
# u, v, p, w, x, t = solver.solve()

# %% 
