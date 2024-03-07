#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
28th Feb, 2024

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
K = torch.tensor([[0.4, 4.0], [3.0, 1.5]])
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
Y == torch.matmul(W, X.reshape(-1)).reshape(2, 2)
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

Z_torch == Z_matmmul
# %%
#Exploring further the formulations of the convolution as matrix multiplication 
#Inspired from this: https://github.com/alisaaalehi/convolution_as_multiplication/tree/main
#and this: https://leimao.github.io/blog/Convolution-Transposed-Convolution-As-Matrix-Multiplication/

#X - input
#Y - convolution output
#K - Kernel
#W - Kernel as a sparse Matrix 

#Convolution : Y = WX
#Transposed Convolution : Y' = W.T Y
#Inverse requires Y' = X 
#Then: X = (W.T W)^-1 Y'


import scipy
import torch.nn.functional as F
import torch

def get_sparse_kernel_matrix(K, h_X, w_X):#Generating the Doubly blocked Toelitz Matrix. 

    # Assuming no channels and stride == 1.
    # Convert the kernel matrix to sparse matrix (dense matrix with lots of zeros in fact).
    # This is a little bit brain-twisting.

    h_K, w_K = K.shape

    h_Y, w_Y = h_X - h_K + 1, w_X - w_K + 1

    W = torch.zeros((h_Y * w_Y, h_X * w_X))
    for i in range(h_Y):
        for j in range(w_Y):
            for ii in range(h_K):
                for jj in range(w_K):
                    W[i * w_Y + j, i * w_X + j + ii * w_X + jj] = K[ii, jj]

    return W


def conv2d_as_matrix_mul(X, K):

    # Assuming no channels and stride == 1.
    # Convert the kernel matrix to sparse matrix (dense matrix with lots of zeros in fact).
    # This is a little bit brain-twisting.

    h_K, w_K = K.shape
    h_X, w_X = X.shape

    h_Y, w_Y = h_X - h_K + 1, w_X - w_K + 1

    W = get_sparse_kernel_matrix(K=K, h_X=h_X, w_X=w_X)

    Y = torch.matmul(W, X.reshape(-1)).reshape(h_Y, w_Y)

    return Y


def conv_transposed_2d_as_matrix_mul(X, K):

    # Assuming no channels and stride == 1.
    # Convert the kernel matrix to sparse matrix (dense matrix with lots of zeros in fact).
    # This is a little bit brain-twisting.

    h_K, w_K = K.shape
    h_X, w_X = X.shape

    h_Y, w_Y = h_X + h_K - 1, w_X + w_K - 1

    # It's like the kernel were applied on the output tensor.
    W = get_sparse_kernel_matrix(K=K, h_X=h_Y, w_X=w_Y)

    # Weight matrix tranposed.
    Y = torch.matmul(W.T, X.reshape(-1)).reshape(h_Y, w_Y)

    return Y

# %% 
K = torch.tensor([[0., 1., 0.],
                      [1., -4., 1.],
                      [0, 1., 0.]])

x = np.linspace(-1, 1, 64) #gridsize
y = x.copy()
xx, yy = np.meshgrid(x, y)

X = torch.tensor(np.exp(-20 *(xx**2 + yy**2)), dtype=torch.float32)

# %% 
#Obtain the sparse matrix
W = get_sparse_kernel_matrix(K, X.shape[0], X.shape[1])

# %% 
#convolution 
conv_matrix_mul = conv2d_as_matrix_mul(X, K)
conv_torch = F.conv2d(X.view(1, 1, X.shape[0], X.shape[1]), K.view(1,1,3,3))[0,0]

# %% 
#transposed convolution 

Y = conv_torch
transp_conv_matrix_mul = conv_transposed_2d_as_matrix_mul(Y, K)
transp_conv_torch = F.conv_transpose2d(Y.view(1, 1, Y.shape[0], Y.shape[1]), K.view(1,1,3,3))[0,0]

# %% 
#Obtaining the pseudoinverse 
# torch.linalg.inv(torch.matmul(W.T,W)) #Try perturbing it and see what we can obtain
pinv = scipy.linalg.pinv(torch.matmul(W.T,W).numpy()) #Moore Penrose Pseudoinvers
pinversed = torch.matmul(torch.tensor(pinv, dtype=torch.float32), X.reshape(-1)).reshape(X.shape[0], X.shape[1])

# %%
#Plotting the conv, transp conv and inverse

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
fig = plt.figure(figsize=(10, 5))

mini = torch.min(X)
maxi = torch.max(X)


# Selecting the axis-X making the bottom and top axes False. 
plt.tick_params(axis='x', which='both', bottom=False, 
                top=False, labelbottom=False) 
  
# Selecting the axis-Y making the right and left axes False 
plt.tick_params(axis='y', which='both', right=False, 
                left=False, labelleft=False) 
  # Remove frame
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)


ax = fig.add_subplot(2,2,1)
pcm =ax.imshow(X, cmap=cm.coolwarm)#, vmin=mini, vmax=maxi)
ax.title.set_text('Input')
# ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(2,2,2)
pcm =ax.imshow(conv_matrix_mul, cmap=cm.coolwarm)#,  vmin=mini, vmax=maxi)
ax.title.set_text('Convolution')
# ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(2,2,3)
pcm =ax.imshow(pinversed, cmap=cm.coolwarm)#, vmin=mini, vmax=maxi)
ax.title.set_text('PseudoInverse')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

ax = fig.add_subplot(2,2,4)
pcm =ax.imshow(transp_conv_matrix_mul, cmap=cm.coolwarm)#,  vmin=mini, vmax=maxi)
ax.title.set_text('Transposed Convolution')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
ax.tick_params(which='both', labelbottom=False, labelleft=False, left=False, bottom=False)

# %%
