
import torch 
import torch.nn as nn
from Utils.VectorConvOps_Spatial import *

class Euler_FV_OS_rhs(nn.Module):#Compressible Navier-Stokes Finite Volume Operator-Splitting right-hand-side.
    def __init__(self, configuration, device):
        super(Euler_FV_OS_rhs, self).__init__()

        self.dx = torch.tensor(configuration['Physics']['dx'], dtype=torch.float32, requires_grad=True).to(device)
        self.dy = torch.tensor(configuration['Physics']['dy'], dtype=torch.float32, requires_grad=True).to(device)
        self.gamma = torch.tensor(5/3, dtype=torch.float32, requires_grad=True).to(device)

        self.gradient = Gradient(scale=1/(self.dx), taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)
        self.laplace = Laplace(scale=1/(self.dx**2), taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)
        self.divergence = Divergence(scale = 1/(self.dx), taylor_order=2, boundary_cond='periodic', device=device, requires_grad=True)

    def forward(self, vars):
        #vars is for a single time instance
        rho = vars[:, 0:1]
        u   = vars[:, 1:2]
        v   = vars[:, 2:3]
        uv  = vars[:, 1:3]
        p   = vars[:, 3:4]
        
        rhs_mass = - rho*self.divergence(u,v) - dot(uv, self.gradient(rho)) 
        rhs_mom = -dot(uv, self.gradient(u)) - dot(uv, self.gradient(v)) + self.laplace(u, v) + (1/rho)*self.gradient(p)            
        rhs_energy = -self.gamma*p*self.divergence(u,v) - dot(uv, self.gradient(rho))        
        
        rhs = torch.cat((rhs_mass, rhs_mom[:, 0:1], rhs_mom[:, 1:2], rhs_energy), dim=1)
        return rhs

    def count_params(self):
        nparams = 0

        for param in self.parameters():
            nparams += param.numel()
        return nparams 
    

from Utils.ConvOps_2d import ConvOperator
class CNS_residuals(nn.Module):
    def __init__(self, boundary, device):
        super(CNS_residuals, self).__init__()

        self.dx = torch.tensor(0.0078, dtype=torch.float32, requires_grad=True).to(device)
        self.dy = torch.tensor(0.0078, dtype=torch.float32, requires_grad=True).to(device)  
        self.dt = torch.tensor(0.05, dtype=torch.float32, requires_grad=True).to(device)
        self.boundary = boundary
        
        #Defining the required Convolutional Operations. 
        self.D_t = ConvOperator(domain='t', order=1)
        self.D_x = ConvOperator(domain='x', order=1)
        self.D_y = ConvOperator(domain='y', order=1)


    def mass(self, vars, boundary=False):
        rho = vars[:, 0:1]
        u   = vars[:, 1:2]
        v   = vars[:, 2:3]
        
        mass_residual = self.D_t(rho) + self.rho*(self.D_x(u) + self.D_y(v)) + u*self.D_x(rho) + v*self.D_y(rho)

        if boundary: 
            return mass_residual
        else:
            return mass_residual[...,1:-1,1:-1,1:-1]
    

