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
run_dict = client.get_runs(['folder.path == /Residuals_UQ/stencil_inversion' and 'filter.tag == Inverse Kernel'], metadata=False)
len(run_dict)
# %%
