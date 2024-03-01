#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
28th Feb, 2023

Exploring the learnd Convolutional Kernels that do the forward mapping from the field to the residuals and the inverse from the residuals to the field. 
The kernels are trained in parallel within the code kernel_learning_stencil.py

Using the wave equation as the base test case for this experiment.
"""

# %% 
# Necessary Imports. 
import os
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from pyDOE import lhs
from tqdm import tqdm 

# %% 
#Using simvue's client API to retrieve the saved kernels. 
from simvue import Client
client = Client()

# %% 
#Obtaining the runs from simvue. 
forward_runs = client.get_runs(['folder.path == /Residuals_UQ/stencil_inversion', 'has tag.Forward Kernel'], metadata=False)
inverse_runs = client.get_runs(['folder.path == /Residuals_UQ/stencil_inversion', 'has tag.Inverse Kernel'], metadata=False)
# %%

#Extracting the Kernels
fks  = [] #forward
for ii in tqdm(range(len(forward_runs))):
    try:
        temp = client.get_artifact(forward_runs[ii]['id'], 'learnt_forward_stencil.npy')
    except:
        pass
    else:
        fks.append(temp)
fks = np.asarray(fks)

bks  = [] #forward
for ii in tqdm(range(len(inverse_runs))):
    try:
        temp = client.get_artifact(inverse_runs[ii]['id'], 'learnt_inverse_stencil.npy')
    except:
        pass
    else:
        bks.append(temp)
bks = np.asarray(bks)
# %%
print("Mean Forward Kernel")
print(np.mean(fks, axis=0))
print()
print("Mean Inverse Kernel")
print(np.mean(bks, axis=0))
# %%
