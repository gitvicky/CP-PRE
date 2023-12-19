"""

Modularised Finite Volutme Constrained MHD Solver - Simulating the Orszag-Tang vortex MHD problem


Code optimised from https://levelup.gitconnected.com/create-your-own-constrained-transport-magnetohydrodynamics-simulation-with-python-276f787f537d

Equations: 

# Domain [0,1] x [0,1]

Writing it out as a class - still work pending. 

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


def getCurl(Az, dx):
	"""
    Calculate the discrete curl
	Az       is matrix of nodal z-component of magnetic potential
	dx       is the cell size
	bx       is matrix of cell face x-component magnetic-field
	by       is matrix of cell face y-component magnetic-field
	"""	
	# directions for np.roll() 
	R = -1   # right/up
	L = 1    # left/down
	
	bx =  ( Az - np.roll(Az,L,axis=1) ) / dx  # = d Az / d y
	by = -( Az - np.roll(Az,L,axis=0) ) / dx  # =-d Az / d x
	
	return bx, by


def getDiv(bx, by, dx):
	"""
    Calculate the discrete curl of each cell
    dx       is the cell size
	bx       is matrix of cell face x-component magnetic-field
	by       is matrix of cell face y-component magnetic-field
	"""	
	# directions for np.roll() 
	R = -1   # right/up
	L = 1    # left/down
	
	divB = (bx - np.roll(bx,L,axis=0) + by - np.roll(by,L,axis=1)) / dx	
	
	return divB
	
	
def getBavg(bx, by):
	"""
    Calculate the volume-averaged magnetic field
	bx       is matrix of cell face x-component magnetic-field
	by       is matrix of cell face y-component magnetic-field
	Bx       is matrix of cell Bx
	By       is matrix of cell By
	"""	
	# directions for np.roll() 
	R = -1   # right/up
	L = 1    # left/down
	
	Bx = 0.5 * ( bx + np.roll(bx,L,axis=0) ) 
	By = 0.5 * ( by + np.roll(by,L,axis=1) ) 
	
	return Bx, By	


def getConserved( rho, vx, vy, P, Bx, By, gamma, vol ):
	"""
    Calculate the conserved variable from the primitive
	rho      is matrix of cell densities
	vx       is matrix of cell x-velocity
	vy       is matrix of cell y-velocity
	P        is matrix of cell Total pressures
	Bx       is matrix of cell Bx
	By       is matrix of cell By
	gamma    is ideal gas gamma
	vol      is cell volume
	Mass     is matrix of mass in cells
	Momx     is matrix of x-momentum in cells
	Momy     is matrix of y-momentum in cells
	Energy   is matrix of energy in cells
	"""
	Mass   = rho * vol
	Momx   = rho * vx * vol
	Momy   = rho * vy * vol
	Energy = ((P-0.5*(Bx**2+By**2))/(gamma-1) + 0.5*rho*(vx**2+vy**2) + 0.5*(Bx**2+By**2)) * vol
	
	return Mass, Momx, Momy, Energy


def getPrimitive( Mass, Momx, Momy, Energy, Bx, By, gamma, vol ):
	"""
    Calculate the primitive variable from the conservative
	Mass     is matrix of mass in cells
	Momx     is matrix of x-momentum in cells
	Momy     is matrix of y-momentum in cells
	Energy   is matrix of energy in cells
	Bx       is matrix of cell Bx
	By       is matrix of cell By
	gamma    is ideal gas gamma
	vol      is cell volume
	rho      is matrix of cell densities
	vx       is matrix of cell x-velocity
	vy       is matrix of cell y-velocity
	P        is matrix of cell Total pressures
	"""
	rho = Mass / vol
	vx  = Momx / rho / vol
	vy  = Momy / rho / vol
	P   = (Energy/vol - 0.5*rho*(vx**2+vy**2) - 0.5*(Bx**2+By**2)) * (gamma-1) + 0.5*(Bx**2+By**2)

	return rho, vx, vy, P


def getGradient(f, dx):
	"""
    Calculate the gradients of a field
	f        is a matrix of the field
	dx       is the cell size
	f_dx     is a matrix of derivative of f in the x-direction
	f_dy     is a matrix of derivative of f in the y-direction
	"""
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left
	
	f_dx = ( np.roll(f,R,axis=0) - np.roll(f,L,axis=0) ) / (2*dx)
	f_dy = ( np.roll(f,R,axis=1) - np.roll(f,L,axis=1) ) / (2*dx)
	
	return f_dx, f_dy


def slopeLimit(f, dx, f_dx, f_dy):
	"""
    Apply slope limiter to slopes
	f        is a matrix of the field
	dx       is the cell size
	f_dx     is a matrix of derivative of f in the x-direction
	f_dy     is a matrix of derivative of f in the y-direction
	"""
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left
	
	f_dx = np.maximum(0., np.minimum(1., ( (f-np.roll(f,L,axis=0))/dx)/(f_dx + 1.0e-8*(f_dx==0)))) * f_dx
	f_dx = np.maximum(0., np.minimum(1., (-(f-np.roll(f,R,axis=0))/dx)/(f_dx + 1.0e-8*(f_dx==0)))) * f_dx
	f_dy = np.maximum(0., np.minimum(1., ( (f-np.roll(f,L,axis=1))/dx)/(f_dy + 1.0e-8*(f_dy==0)))) * f_dy
	f_dy = np.maximum(0., np.minimum(1., (-(f-np.roll(f,R,axis=1))/dx)/(f_dy + 1.0e-8*(f_dy==0)))) * f_dy
	
	return f_dx, f_dy


def extrapolateInSpaceToFace(f, f_dx, f_dy, dx):
	"""
    Calculate the gradients of a field
	f        is a matrix of the field
	f_dx     is a matrix of the field x-derivatives
	f_dy     is a matrix of the field y-derivatives
	dx       is the cell size
	f_XL     is a matrix of spatial-extrapolated values on `left' face along x-axis 
	f_XR     is a matrix of spatial-extrapolated values on `right' face along x-axis 
	f_YL     is a matrix of spatial-extrapolated values on `left' face along y-axis 
	f_YR     is a matrix of spatial-extrapolated values on `right' face along y-axis 
	"""
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left
	
	f_XL = f - f_dx * dx/2
	f_XL = np.roll(f_XL,R,axis=0)
	f_XR = f + f_dx * dx/2
	
	f_YL = f - f_dy * dx/2
	f_YL = np.roll(f_YL,R,axis=1)
	f_YR = f + f_dy * dx/2
	
	return f_XL, f_XR, f_YL, f_YR


def applyFluxes(F, flux_F_X, flux_F_Y, dx, dt):
	"""
    Apply fluxes to conserved variables
	F        is a matrix of the conserved variable field
	flux_F_X is a matrix of the x-dir fluxes
	flux_F_Y is a matrix of the y-dir fluxes
	dx       is the cell size
	dt       is the timestep
	"""
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left
	
	# update solution
	F += - dt * dx * flux_F_X
	F +=   dt * dx * np.roll(flux_F_X,L,axis=0)
	F += - dt * dx * flux_F_Y
	F +=   dt * dx * np.roll(flux_F_Y,L,axis=1)
	
	return F


def constrainedTransport(bx, by, flux_By_X, flux_Bx_Y, dx, dt):
	"""
    Apply fluxes to face-centered magnetic fields in a constrained transport manner
	bx        is matrix of cell face x-component magnetic-field
	by        is matrix of cell face y-component magnetic-field
	flux_By_X is a matrix of the x-dir fluxes of By
	flux_Bx_Y is a matrix of the y-dir fluxes of Bx
	dx        is the cell size
	dt        is the timestep
	"""
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left
	
	# update solution
	# Ez at top right node of cell = avg of 4 fluxes
	Ez = 0.25 * ( -flux_By_X - np.roll(flux_By_X,R,axis=1) + flux_Bx_Y + np.roll(flux_Bx_Y,R,axis=0) )
	dbx, dby = getCurl(-Ez, dx)
	
	bx += dt * dbx
	by += dt * dby
	
	return bx, by 


def getFlux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, Bx_L, Bx_R, By_L, By_R, gamma):
	"""
    Calculate fluxed between 2 states with local Lax-Friedrichs/Rusanov rule 
	rho_L        is a matrix of left-state  density
	rho_R        is a matrix of right-state density
	vx_L         is a matrix of left-state  x-velocity
	vx_R         is a matrix of right-state x-velocity
	vy_L         is a matrix of left-state  y-velocity
	vy_R         is a matrix of right-state y-velocity
	P_L          is a matrix of left-state  Total pressure
	P_R          is a matrix of right-state Total pressure
	Bx_L         is a matrix of left-state  x-magnetic-field
	Bx_R         is a matrix of right-state x-magnetic-field
	By_L         is a matrix of left-state  y-magnetic-field
	By_R         is a matrix of right-state y-magnetic-field
	gamma        is the ideal gas gamma
	flux_Mass    is the matrix of mass fluxes
	flux_Momx    is the matrix of x-momentum fluxes
	flux_Momy    is the matrix of y-momentum fluxes
	flux_Energy  is the matrix of energy fluxes
	"""
	
	# left and right energies
	en_L = (P_L - 0.5*(Bx_L**2+By_L**2))/(gamma-1) + 0.5*rho_L*(vx_L**2+vy_L**2) + 0.5*(Bx_L**2+By_L**2)
	en_R = (P_R - 0.5*(Bx_R**2+By_R**2))/(gamma-1) + 0.5*rho_R*(vx_R**2+vy_R**2) + 0.5*(Bx_R**2+By_R**2)

	# compute star (averaged) states
	rho_star  = 0.5*(rho_L + rho_R)
	momx_star = 0.5*(rho_L * vx_L + rho_R * vx_R)
	momy_star = 0.5*(rho_L * vy_L + rho_R * vy_R)
	en_star   = 0.5*(en_L + en_R)
	Bx_star   = 0.5*(Bx_L + Bx_R)
	By_star   = 0.5*(By_L + By_R)
	
	P_star = (gamma-1)*(en_star - 0.5*(momx_star**2+momy_star**2)/rho_star - 0.5*(Bx_star**2+By_star**2)) + 0.5*(Bx_star**2+By_star**2)
	
	# compute fluxes (local Lax-Friedrichs/Rusanov)
	flux_Mass   = momx_star
	flux_Momx   = momx_star**2/rho_star + P_star - Bx_star * Bx_star
	flux_Momy   = momx_star * momy_star/rho_star - Bx_star * By_star
	flux_Energy = (en_star+P_star) * momx_star/rho_star - Bx_star * (Bx_star*momx_star + By_star*momy_star) / rho_star
	flux_By     = (By_star * momx_star - Bx_star * momy_star) / rho_star
	
	# find wavespeeds
	c0_L = np.sqrt( gamma*(P_L-0.5*(Bx_L**2+By_L**2))/rho_L )
	c0_R = np.sqrt( gamma*(P_R-0.5*(Bx_R**2+By_R**2))/rho_R )
	ca_L = np.sqrt( (Bx_L**2+By_L**2)/rho_L )
	ca_R = np.sqrt( (Bx_R**2+By_R**2)/rho_R )
	cf_L = np.sqrt( 0.5*(c0_L**2+ca_L**2) + 0.5*np.sqrt((c0_L**2+ca_L**2)**2) )
	cf_R = np.sqrt( 0.5*(c0_R**2+ca_R**2) + 0.5*np.sqrt((c0_R**2+ca_R**2)**2) )
	C_L = cf_L+ np.abs(vx_L)
	C_R = cf_R + np.abs(vx_R)
	C = np.maximum( C_L, C_R )
	
	# add stabilizing diffusive term
	flux_Mass   -= C * 0.5 * (rho_L - rho_R)
	flux_Momx   -= C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
	flux_Momy   -= C * 0.5 * (rho_L * vy_L - rho_R * vy_R)
	flux_Energy -= C * 0.5 * ( en_L - en_R )
	flux_By     -= C * 0.5 * ( By_L - By_R )

	return flux_Mass, flux_Momx, flux_Momy, flux_Energy, flux_By

# %% 

class constrainedTransport_MHD:
	def __init__(self, N, L, tEnd, a=1, b=1):

		self.N = N
		self.boxsize = L
		self.gamma = 5/3
		self.courant_fac = 0.4
		self.t = 0 
		self.tEnd = tEnd
		self.useSlopeLimiting = True 

		self.tOut = 0.01
		self.outputCount = 1

		#Mesh
		self.dx = self.boxsize / self.N
		self.vol = self.dx**2
		self.xlin = np.linspace(0.5*self.dx, self.boxsize - 0.5*self.dx, self.N)
		self.Y, self.X= np.meshgrid(self.xlin, self.xlin)
		self.xlin_node = np.linspace(self.dx, self.boxsize, self.N)
		self.Yn, self.Xn = np.meshgrid(self.xlin_node, self.xlin_node)

		#Generate Initial Conditions 
		self.rho = (self.gamma**2 / (4*np.pi)) * np.ones(self.X.shape)
		self.vx = -np.sin(a*2*np.pi*self.Y)
		self.vy = np.sin(b*2*np.pi*self.X)
		self.P = (self.gamma / (4*np.pi)) * np.ones(self.X.shape)

		#Magnetic Field IC
		#Az is at the top right-node of each cell. §
		self.Az  = np.cos(4*np.pi*self.X) / (4*np.pi*np.sqrt(4*np.pi)) + np.cos(2*np.pi*self.Y) / (2*np.pi*np.sqrt(4*np.pi))
		self.bx, self.by = getCurl(self.Az, self.dx)
		self.Bx, self.By = getBavg(self.bx, self.by)

		#Addign Mag Pressire to Total Pressure 
		self.P = self.P + 0.5*(self.Bx**2 + self.By**2)

		#Get Conserved Variables
		self.Mass, self.Momx, self.Momy, self.Energy = getConserved(self.rho, self.vx, self.vy, self.P, self.Bx, self.By, self.gamma, self.vol)

	def solve(self):
		err = 0 
		rho_list = []
		u_list = []
		v_list = []
		p_list = []
		bx_list = []
		by_list = []

		while self.t < self.tEnd:

			#Get Primitive Variables 
			self.Bx, self.By = getBavg(self.bx, self.by)
			self.rho, self.vx, self.vy, self.P = getPrimitive(self.Mass, self.Momx, self.Momy, self.Energy, self.Bx, self.By, self.gamma, self.vol)

			#Get timestep (CFL) = dx /max signal speed
			c0 = np.sqrt(self.gamma*(self.P-0.5*(self.Bx**2 + self.By**2)) / self.rho)
			ca = np.sqrt((self.Bx**2 + self.By**2))
			cf = np.sqrt(0.5 * (c0**2 + ca**2) + 0.5*np.sqrt((c0**2 + ca**2)**2))
			self.dt = self.courant_fac * np.min(self.dx / (cf + np.sqrt(self.vx**2 + self.vy**2)))
			
			if self.t + self.dt > self.outputCount*self.tOut:
				self.dt = self.outputCount*self.tOut - self.t

			self.rho_dx, self.rho_dy = getGradient(self.rho, self.dx)
			self.vx_dx, self.vx_dy = getGradient(self.vx, self.dx)
			self.vy_dx, self.vy_dy = getGradient(self.vy, self.dx)
			self.P_dx, self.P_dy = getGradient(self.P, self.dx)
			self.Bx_dx, self.Bx_dy = getGradient(self.Bx, self.dx)
			self.By_dx, self.By_dy = getGradient(self.By, self.dx)

			#slope limit gradients
			if self.useSlopeLimiting:
				self.rho_dx, self.rho_dy = slopeLimit(self.rho, self.dx, self.rho_dx, self.rho_dy)
				self.vx_dx, self.vx_dy = slopeLimit(self.vx, self.dx, self.vx_dx, self.vx_dy)
				self.vy_dx, self.vy_dy = slopeLimit(self.vy, self.dx, self.vy_dx, self.vy_dy)
				self.P_dx, self.P_dy = slopeLimit(self.P, self.dx, self.P_dx, self.P_dy)
				self.Bx_dx, self.Bx_dy = slopeLimit(self.Bx, self.dx, self.Bx_dx, self.Bx_dy)
				self.By_dx, self.By_dy = slopeLimit(self.By, self.dx, self.By_dx, self.By_dy)

			#Extrapolate half-speed in time
			rho_prime = self.rho - 0.5*self.dt * (self.vx * self.rho_dx + self.rho * self.vx_dx + self.vy * self.rho_dy + self.rho * self.vy_dy)
			vx_prime = self.vx - 0.5*self.dt * (self.vx * self.vx_dx + self.vy*self.vx_dy + (1/self.rho) * self.P_dx - (2*self.Bx/self.rho) * self.Bx_dx - (self.By/self.rho) * self.Bx_dy - (self.Bx/self.rho) * self.By_dy)
			vy_prime = self.vy - 0.5*self.dt * (self.vx*self.vy_dx + self.vy*self.vy_dy + (1/self.rho) * self.P_dy - (2*self.By/self.rho) * self.By_dy - (self.Bx/self.rho) * self.By_dx - (self.By/self.rho) * self.Bx_dx)
			P_prime = self.P - 0.5*self.dt * ((self.gamma*(self.P - 0.5*(self.Bx**2 + self.By**2)) + self.By**2)*self.vx_dx - self.vx_dx - self.Bx*self.By*self.vy_dx + self.vx*self.P_dx + (self.gamma-2) * (self.Bx*self.vx + self.By*self.vy)*self.Bx_dx - self.By*self.Bx*self.vx_dy + (self.gamma*(self.P - 0.5*(self.Bx**2 + self.By**2))+self.Bx**2)*self.vy_dy + self.vy*self.P_dy + (self.gamma-2)*(self.Bx*self.vx + self.By*self.vy)*self.By_dy)
			Bx_prime = self.Bx - 0.5*self.dt*(-self.By*self.vx_dy + self.Bx * self.vy_dy + self.vy *self.Bx_dy - self.vx * self.By_dy)
			By_prime = self.By - 0.5*self.dt*(self.By*self.vx_dx - self.Bx * self.vy_dx - self.vy * self.Bx_dx + self.vx * self.By_dx)

			#Extrapolate in space to face centres
			rho_XL, rho_XR,rho_YL, rho_YR = extrapolateInSpaceToFace(rho_prime, self.rho_dx, self.rho_dy, self.dx)
			vx_XL, vx_XR,vx_YL, vx_YR = extrapolateInSpaceToFace(vx_prime, self.vx_dx, self.vx_dy, self.dx)
			vy_XL, vy_XR,vy_YL, vy_YR = extrapolateInSpaceToFace(vy_prime, self.vy_dx, self.vy_dy, self.dx)
			P_XL, P_XR,P_YL, P_YR = extrapolateInSpaceToFace(P_prime, self.P_dx, self.P_dy, self.dx)
			Bx_XL, Bx_XR,Bx_YL, Bx_YR = extrapolateInSpaceToFace(Bx_prime, self.Bx_dx, self.Bx_dy, self.dx)
			By_XL, By_XR,By_YL, By_YR = extrapolateInSpaceToFace(By_prime, self.By_dx, self.By_dy, self.dx)

			#Compute fluxes (local Lax-Friedrichs/Rusanov)
			flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X, flux_By_X = getFlux(rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, P_XL, P_XR, Bx_XL, Bx_XR, By_XL, By_XR, self.gamma)
			flux_Mass_y, flux_Momy_Y, flux_Momx_Y, flux_Energy_Y, flux_Bx_Y = getFlux(rho_YL, rho_YR, vx_YL, vx_YR, vy_YL, vy_YR, P_YL, P_YR, Bx_YL, Bx_XR, By_XL, By_YR, self.gamma)

			#Update Solution 

			self.Mass = applyFluxes(self.Mass, flux_Mass_X, flux_Mass_y, self.dx, self.dt)
			self.Momx = applyFluxes(self.Momx, flux_Momx_X, flux_Momx_Y, self.dx, self.dt)
			self.Momy = applyFluxes(self.Momy, flux_Momy_X, flux_Momy_Y, self.dx, self.dt)
			self.Energy = applyFluxes(self.Energy, flux_Energy_X, flux_Energy_Y, self.dx, self.dt)
			self.bx, self.by = constrainedTransport(self.bx, self.by, flux_By_X, flux_Bx_Y, self.dx, self.dt)

			#Update time
			self.t += self.dt

			#Check Div B
			divB = getDiv(self.bx, self.by, self.dx)
			mean_divB = np.mean((np.abs(divB)))
			print("t= ", self.t, ", mean |divB| = ", mean_divB)

			if divB > 1 : 
				err = 1

			rho_list.append(self.rho)
			u_list.append(self.vx)
			v_list.append(self.vy)
			p_list.append(self.P)
			bx_list.append(self.bx)
			by_list.append(self.by)


		return np.asarray(rho_list), np.asarray(u_list), np.asarray(v_list), np.asarray(p_list), np.asarray(bx_list), np.asarray(by_list), err

# %% 
solver= constrainedTransport_MHD(128, 1.0, 0.5, 1.0, 1.0, 1.0) # N, L, tEnd, a, b, c
rho, u, v, p, bx, by = solver.solve()
