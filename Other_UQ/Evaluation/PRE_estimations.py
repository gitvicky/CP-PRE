# %%
import torch 
from Utils.ConvOps_2d import ConvOperator

class PRE_Wave():
    def __init__(self, dt, dx, c=1.0):

        D_tt = ConvOperator('t', 2)#, scale=alpha)
        D_xx_yy = ConvOperator(('x','y'), 2)#, scale=beta)
        self.D = ConvOperator()
        c = torch.tensor(c, dtype=torch.float32)
        self.D.kernel = D_tt.kernel - (c*dt/dx)**2 * D_xx_yy.kernel
    
    def residual(self, uu, boundary=False):
        # Residual
        uu = uu[:,0]
        res = self.D(uu)
        if boundary:
            return res
        else: 
            return res[...,1:-1,1:-1,1:-1]


class PRE_NS():
    def __init__(self, dt, dx, dy):

        self.dt = dt 
        self.dx = dx 
        self.dy = dy 
        
        #Defining the required Convolutional Operations. 
        self.D_t = ConvOperator(domain='t', order=1)#, scale=alpha)
        self.D_x = ConvOperator(domain='x', order=1)#, scale=beta) 
        self.D_y = ConvOperator(domain='y', order=1)#, scale=beta)
        self.D_x_y = ConvOperator(domain=('x', 'y'), order=1)#, scale=beta)
        self.D_xx_yy = ConvOperator(domain=('x','y'), order=2)#, scale=gamma)

                
    #Momentum 
    def residual(self, vars, boundary=False):
        u, v, p = vars[:,0], vars[:, 1], vars[:, 2]
        nu = 0.001

        res_x = self.D_t(u)*self.dx*self.dy + u*self.D_x(u)*self.dt*self.dy + v*self.D_y(u)*self.dt*self.dx - nu*self.D_xx_yy(u)*self.dt + self.D_x(p)*self.dt*self.dy
        res_y = self.D_t(v)*self.dx*self.dy + u*self.D_x(v)*self.dt*self.dx + v*self.D_y(v)*self.dt*self.dy - nu*self.D_xx_yy(v)*self.dt + self.D_y(p)*self.dt*self.dx

        if boundary:
            return res_x + res_y
        else: 
            return res_x[...,1:-1,1:-1,1:-1] + res_y[...,1:-1,1:-1,1:-1]
        


class PRE_MHD():
    def __init__(self, dt, dx, dy):

        self.dt = dt 
        self.dx = dx 
        self.dy = dy 
        
        #Defining the required Convolutional Operations. 
        self.D_t = ConvOperator(domain='t', order=1)#, scale=alpha)
        self.D_x = ConvOperator(domain='x', order=1)#, scale=beta) 
        self.D_y = ConvOperator(domain='y', order=1)#, scale=beta)
        self.D_x_y = ConvOperator(domain=('x', 'y'), order=1)#, scale=beta)
        self.D_xx_yy = ConvOperator(domain=('x','y'), order=2)#, scale=gamma)


    #Energy
    def residual(self, vars, boundary=False):
        gamma = 5/3
        rho, u, v, p, Bx, By = vars[:, 0], vars[:, 1], vars[:, 2], vars[:, 3], vars[:, 4], vars[:, 5]
        p_gas = p - 0.5*(Bx**2 + By**2)

        res = self.D_t(rho) + u*self.D_x(p) + v*self.D_y(p) + (gamma-2)*(u*Bx+v*By)*(self.D_x(Bx) + self.D_y(By)) + (gamma*p_gas+By**2)*self.D_x(u) + (gamma*p_gas+Bx**2)*self.D_y(v)- Bx*By*(self.D_y(u) + self.D_x(v))
        
        if boundary:
            return res
        else: 
            return res[...,1:-1,1:-1,1:-1]