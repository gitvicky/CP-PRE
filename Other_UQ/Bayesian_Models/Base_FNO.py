# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22 Apr, 2024
author: @vgopakum, @zongyli

Multivariate-FNO in 2D
"""
# %% 
################################################################
# FNO - Code worked and modified with Caltech (Zongyi et al.)
################################################################
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import operator
from functools import reduce
from functools import partial
from collections import OrderedDict


# %% 
################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, num_vars, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_vars = num_vars
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.num_vars, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.num_vars, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bivxy,iovxy->bovxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, self.num_vars, x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP2d, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)
        self.activation = F.gelu

    def forward(self, x):
        x = self.mlp1(x)
        x = self.activation(x)
        x = self.mlp2(x)
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, vars, width):
        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.vars = vars
        self.width = width

        self.conv = SpectralConv2d(self.width, self.width, self.vars, self.modes1, self.modes2)
        self.mlp = MLP2d(self.width, self.width, self.width)
        self.w = nn.Conv3d(self.width, self.width, 1)
        self.b = nn.Conv3d(2, self.width, 1)

        self.activation = F.gelu

    def forward(self, x, grid):
        x1 = self.conv(x)
        x1 = self.mlp(x1)
        x2 = self.w(x)
        x3 = self.b(grid)
        x = x1 + x2 + x3
        x = self.activation(x)
        return x

# %%

class FNO_multi2d(nn.Module):
    def __init__(self, T_in, step, modes1, modes2, num_vars, width_time, width_vars=0, grid='arbitrary'):
        super(FNO_multi2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous T_in timesteps + 2 locations (u(t-T_in, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=x_discretistion, y=y_discretisation, c=T_in)
        output: the solution of the next timestep
        output shape: (batchsize, x=x_discretisation, y=y_discretisatiob, c=step)
        """

        self.T_in = T_in
        self.step = step
        self.modes1 = modes1
        self.modes2 = modes2
        self.num_vars = num_vars
        self.width_vars = width_vars
        self.width_time = width_time
        self.grid = grid

        self.fc0_time = nn.Linear(self.T_in + 2, self.width_time) #+2 for the spatial discretisations in 2D

        # self.padding = 8 # pad the domain if input is non-periodic

        self.f0 = FNO2d(self.modes1, self.modes2, self.num_vars, self.width_time)
        self.f1 = FNO2d(self.modes1, self.modes2, self.num_vars, self.width_time)
        self.f2 = FNO2d(self.modes1, self.modes2, self.num_vars, self.width_time)
        self.f3 = FNO2d(self.modes1, self.modes2, self.num_vars, self.width_time)
        self.f4 = FNO2d(self.modes1, self.modes2, self.num_vars, self.width_time)
        self.f5 = FNO2d(self.modes1, self.modes2, self.num_vars, self.width_time)

        # self.norm = nn.InstanceNorm2d(self.width)
        self.norm = nn.Identity()

        self.fc1_time = nn.Linear(self.width_time, 256)
        self.fc2_time = nn.Linear(256, self.step)

        self.activation = torch.nn.GELU()

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0_time(x)
        x = x.permute(0, 4, 1, 2, 3)
        grid = grid.permute(0, 4, 1, 2, 3)

        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x0 = self.f0(x, grid)
        x = self.f1(x0, grid)
        x = self.f2(x, grid) + x0
        x1 = self.f3(x, grid)
        x = self.f4(x1, grid)
        x = self.f5(x, grid) + x1

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic

        x = x.permute(0, 2, 3, 4, 1)

        x = self.fc1_time(x)
        x = self.activation(x)
        x = self.fc2_time(x)

        return x
    
    def get_grid(self, shape, device):
        
        batchsize, self.num_vars, size_x, size_y = shape[0], shape[1], shape[2], shape[3]         
        if self.grid == 'arbitrary':
                gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
                gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        else:
            gridx = self.grid[0]
            gridy = self.grid[1]
    
        gridx = gridx.reshape(1, 1, size_x, 1, 1).repeat([batchsize, self.num_vars, 1, size_y, 1])
        gridy = gridy.reshape(1, 1, 1, size_y, 1).repeat([batchsize, self.num_vars, size_x, 1, 1])

        return torch.cat((gridx, gridy), dim=-1).to(device)

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c
# %%
#Example Usage
# model = FNO_multi(T_in=20, step=5, modes1=8, modes2=8, num_vars=1, width_time=32, width_vars=0)
# ins = torch.randn(100,1,64,64,20) #BS, num_vars, Nx, Ny, T_in
# outs = model(ins)
# %% 
# 1D FNO 

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, num_vars, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_vars = num_vars
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.num_vars, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bivx,iovx->bovx", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, self.num_vars, x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :, :self.modes1] = \
            self.compl_mul2d(x_ft[:, :, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class MLP1d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP1d, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
        self.activation = F.gelu

    def forward(self, x):
        x = self.mlp1(x)
        x = self.activation(x)
        x = self.mlp2(x)
        return x


class FNO1d(nn.Module):
    def __init__(self, modes1, vars, width):
        super(FNO1d, self).__init__()

        self.modes1 = modes1
        self.vars = vars
        self.width = width

        self.conv = SpectralConv1d(self.width, self.width, self.vars, self.modes1)
        self.mlp = MLP1d(self.width, self.width, self.width)
        self.w = nn.Conv2d(self.width, self.width, 1)
        self.b = nn.Conv2d(1, self.width, 1)

        self.activation = F.gelu

    def forward(self, x, grid):
        x1 = self.conv(x)
        x1 = self.mlp(x1)
        x2 = self.w(x)
        x3 = self.b(grid)
        x = x1 + x2 + x3
        x = self.activation(x)
        return x

class FNO_multi1d(nn.Module):
    def __init__(self, T_in, step, modes1, num_vars, width_time, width_vars=0, grid='arbitrary'):
        super(FNO_multi1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous T_in timesteps + 2 locations (u(t-T_in, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=x_discretistion, y=y_discretisation, c=T_in)
        output: the solution of the next timestep
        output shape: (batchsize, x=x_discretisation, y=y_discretisatiob, c=step)
        """

        self.T_in = T_in
        self.step = step
        self.modes1 = modes1
        self.num_vars = num_vars
        self.width_vars = width_vars
        self.width_time = width_time
        self.grid = grid

        self.fc0_time = nn.Linear(self.T_in + 1, self.width_time) #+2 for the spatial discretisations in 2D

        # self.padding = 8 # pad the domain if input is non-periodic

        self.f0 = FNO1d(self.modes1, self.num_vars, self.width_time)
        self.f1 = FNO1d(self.modes1, self.num_vars, self.width_time)
        self.f2 = FNO1d(self.modes1, self.num_vars, self.width_time)
        self.f3 = FNO1d(self.modes1, self.num_vars, self.width_time)
        self.f4 = FNO1d(self.modes1, self.num_vars, self.width_time)
        self.f5 = FNO1d(self.modes1, self.num_vars, self.width_time)

        # self.norm = nn.InstanceNorm2d(self.width)
        self.norm = nn.Identity()

        self.fc1_time = nn.Linear(self.width_time, 256)
        self.fc2_time = nn.Linear(256, self.step)

        self.activation = torch.nn.GELU()
    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x = self.fc0_time(x)
        x = x.permute(0, 3, 1, 2)
        grid = grid.permute(0, 3, 1, 2)

        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x0 = self.f0(x, grid)
        x = self.f1(x0, grid)
        x = self.f2(x, grid) + x0
        x1 = self.f3(x, grid)
        x = self.f4(x1, grid)
        x = self.f5(x, grid) + x1

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic

        x = x.permute(0, 2, 3, 1)

        x = self.fc1_time(x)
        x = self.activation(x)
        x = self.fc2_time(x)

        return x
    
    def get_grid(self, shape, device):
        
        batchsize, self.num_vars, size_x = shape[0], shape[1], shape[2]   
        if self.grid == 'arbitrary':
                gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        else:
            gridx = self.grid[0]
    
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, self.num_vars, 1, 1])

        return gridx.to(device)

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c
    
# %%
# #Example Usage
# model = FNO_multi1d(T_in=20, step=5, modes1=8, num_vars=1, width_time=32, width_vars=0)
# ins = torch.randn(100,1,64,20) #BS, num_vars, Nx, Ny, T_in
# outs = model(ins)
# %%
