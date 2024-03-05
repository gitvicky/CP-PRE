#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
28th Feb, 2023

Exploring the transformations and mathematical formulations of a Convolution and its Trasnposed Operation. 
Understanding further from this: https://d2l.ai/chapter_computer-vision/transposed-conv.html
"""

# %% 
# Necessary Imports. 
import os
import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn
import torch.nn.functional as F


# %% 
#Setting up a convolution 

X = torch.arange(9.0).reshape(3, 3)
K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = F.conv2d(X.view(1,1,3,3), K.view(1,1,2,2))


# %%
#Writing up the Kernel as a sparse matrix 
def kernel2matrix(K):
    k, W = torch.zeros(5), torch.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W

W = kernel2matrix(K)
print(W)

# %%
#Obtaining the Convolution as a multiplication between the sparse weight matrix and the input
Y = torch.matmul(W, X.reshape(-1)).reshape(2, 2)

# %% 
# Transposed Convolution hand-written out 
def trans_conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y

Z_handwritten = trans_conv(Y, K)
Z_torch = F.conv_transpose2d(Y.view(1,1,2,2), K.view(1,1,2,2))
Z_matmmul = torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3)
# %%
