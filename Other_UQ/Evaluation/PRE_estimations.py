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
