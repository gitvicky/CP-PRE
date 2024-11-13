# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 6 Jan 2023
author: @vgopakum
Utilities required for training neural-pde surrogate models.
"""
import numpy as np 
import torch 
import torch.nn as nn 
import torch.functional as F 


# %%
##################################
# Normalisation Functions
##################################

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.01):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps  # T*batch*n
                mean = self.mean[:, sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.01):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=-1.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low) / (mymax - mymin)
        self.b = -self.a * mymax + high

    def encode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = self.a * x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = (x - self.b) / self.a
        x = x.view(s)
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()


# normalization, rangewise but each variable at a time.
class MinMax_Normalizer_variable(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(MinMax_Normalizer_variable, self).__init__()
        
        self.num_vars = x.shape[1]
        aa = []
        bb = []
        
        for ii in range(self.num_vars):
            min_u = torch.min(x[:, ii, :, :, :])
            max_u = torch.max(x[:, ii, :, :, :])
            
            aa.append((high - low) / (max_u - min_u))
            bb.append( -aa[ii] * max_u + high)
        
        self.a = torch.tensor(aa)
        self.b = torch.tensor(bb)

    def encode(self, x):
        for ii in range(self.num_vars):
            x[:, ii] = self.a[ii] * x[:, ii] + self.b[ii] 
        return x

    def decode(self, x):
        for ii in range(self.num_vars):
            x[:, ii] =  (x[:, ii] - self.b[ii])  /  self.a[ii] 
        return x
    
    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()


    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()


class LogNormalizer(object):
    def __init__(self, x,  low=0.0, high=1.0, eps=0.01):
        super(LogNormalizer, self).__init__()

        self.num_vars = x.shape[1]
        aa = []
        bb = []
        
        for ii in range(self.num_vars):
            min_u = torch.min(x[:, ii, :, :, :])
            max_u = torch.max(x[:, ii, :, :, :])
            
            aa.append((high - low) / (max_u - min_u))
            bb.append( -aa[ii] * max_u + high)
        
        self.a = torch.tensor(aa)
        self.b = torch.tensor(bb)
        
        self.eps = eps

    def encode(self, x):
        for ii in range(self.num_vars):
            x[:, ii] = self.a[ii] * x[:, ii] + self.b[ii] 

        x = torch.log(x + 1 + self.eps)

        return x

    def decode(self, x):
        for ii in range(self.num_vars):
            x[:, ii] =  (x[:, ii] - self.b[ii])  /  self.a[ii] 
        x = torch.exp(x) - 1 - self.eps
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()


    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()


#normalization, rangewise but across the full domain 
class MinMax_Normalizer(object):
    def __init__(self, x, low=-1.0, high=1.0):
        super(MinMax_Normalizer, self).__init__()
        mymin = torch.min(x)
        mymax = torch.max(x)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()


#normalization, Identity - does nothing
class Identity(object):
    def __init__(self, x, low=-1.0, high=1.0):
        super(Identity, self).__init__()
        self.a = torch.tensor(0)
        self.b = torch.tensor(0)

    def encode(self, x):
        return x 

    def decode(self, x):
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()

# %% 
##################################
# Loss Functions
##################################

# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):

        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss


# Fully Connected Network or a Multi-Layer Perceptron
class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_layers, num_neurons, activation=torch.tanh):
        super(MLP, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.num_neurons = num_neurons

        self.act_func = activation

        self.layers = nn.ModuleList()

        self.layer_input = nn.Linear(self.in_features, self.num_neurons)

        for ii in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.num_neurons, self.num_neurons))
        self.layer_output = nn.Linear(self.num_neurons, self.out_features)

    def forward(self, x):
        x_temp = self.act_func(self.layer_input(x))
        for dense in self.layers:
            x_temp = self.act_func(dense(x_temp))
        x_temp = self.layer_output(x_temp)
        return x_temp
    

#Sampling equidistantly from a 2D grid.
def sample_equidistant(grid, num_samples):
   
    #    Sample values from a 2D NumPy grid in an equidistant manner.

    # Args:
    #     grid (numpy.ndarray): A 3D NumPy array representing the num_sim, x_grid, y_grid
    #     num_samples (int): The total number of samples to retrieve equidistantly along each axis.

    # Returns:
    #     numpy.ndarray: A 3D NumPy array containing the sampled values from the equidistant points

    sims, height, width = grid.shape
    num_samples = int(np.sqrt(num_samples))
    # Generate equidistant x-coordinates
    x_coords = np.linspace(0, width - 1, num_samples, dtype=int)
    
    # Generate equidistant y-coordinates
    y_coords = np.linspace(0, height - 1, num_samples, dtype=int)
    
    # Create a meshgrid of the x and y coordinates
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    # Flatten the meshgrid to get the indices
    indices = np.vstack((yy.flatten(), xx.flatten())).T
    
    # Sample the grid using the indices
    samples = grid[:, indices[:, 0], indices[:, 1]]
    
    return samples
