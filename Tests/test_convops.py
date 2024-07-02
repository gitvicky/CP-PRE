#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2nd July, 2024
ConvOps Framework Testing
"""

# %% 
import numpy as np 
import torch 
import torch.nn.functional as F
from matplotlib import pyplot as plt 
# %% 
#Generating the signal - 2D Gaussian 
grid_size = 128
x = np.linspace(-1, 1, grid_size)
dx = x[1] -  x[0]
y = x.copy()
xx, yy = np.meshgrid(x, y)

uu = np.exp(-50*(xx**2 + yy**2))
field = torch.Tensor(uu)
field   = field.view(1, 1, grid_size, grid_size)

# %% 
import sys
sys.path.append("..")

#Using the basic ConvOps settings 
from Utils.ConvOps_2d import ConvOperator
D_xx_yy = ConvOperator(('x','y'), 2)#, scale=beta)
deriv_basic = D_xx_yy(field)

#Using Laplace Class
from Utils.VectorConvOps import Laplace
D_laplace = Laplace()
deriv_laplace = D_laplace(field)

# %% 
from Utils.plot_tools import subplots_2d

#Plotting the fields, prediction, abs error and the residual

values = [deriv_basic,
          deriv_laplace]

titles = ['ConvOps', 'Laplace']

subplots_2d(values, titles)

# %%
#Using the basic ConvOps settings 
D_x = ConvOperator(('x'), 2)#, scale=beta)
D_y = ConvOperator(('y'), 2)#, scale=beta)
deriv_div = D_x(field) +  D_y(field)
deriv_y = D_y(field)

#Using Divergence Class
from Utils.ConvOps_2d import Divergence
D_laplace = Divergence()
deriv_divs = D_laplace(field)

# %% 
from Utils.plot_tools import subplots_2d

#Plotting the fields, prediction, abs error and the residual

values = [deriv_div,
          deriv_divs]

titles = ['ConvOps', 'Divergence']

subplots_2d(values, titles)

# %%
#Using Gradient  Class
from Utils.ConvOps_2d import Gradient
D_grad = Gradient()
g_x, g_y = D_grad(field)

# %% 
from Utils.plot_tools import subplots_2d

#Plotting the fields, prediction, abs error and the residual

values = [ D_x(field),
           D_y(field), 
           g_x, 
           g_y]

titles = ['convops_x', 'convops_y', 'G_x', 'G_y']

subplots_2d(values, titles)

# %% 