
#%% 
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

laplacian_stencil_2nd = torch.tensor([[0, 1, 0],
                                      [1, -4, 1],
                                      [0, 1, 0]], dtype=torch.float32)


laplacian_stencil_4th = torch.tensor([[0, 0, -1/12, 0, 0],
                                      [0, 0, 4/3, 0, 0],
                                      [-1/12, 4/3, -5/2, 4/3, -1/12],
                                      [0, 0, 4/3, 0, 0],
                                      [0, 0, -1/12, 0, 0]], dtype=torch.float32)


laplacian_stencil_6th = torch.tensor([[0, 0, 0, 1/90, 0, 0, 0],
                                      [0, 0, 0, -3/20, 0, 0, 0],
                                      [0, 0, 0, 3/2, 0, 0, 0],
                                      [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90],
                                      [0, 0, 0, 3/2, 0, 0, 0],
                                      [0, 0, 0, -3/20, 0, 0, 0],
                                      [0, 0, 0, 1/90, 0, 0, 0]], dtype=torch.float32)

grid_size = 128
x = np.linspace(-1, 1, grid_size)
dx = x[1] -  x[0]
y = x.copy()
xx, yy = np.meshgrid(x, y)
uu = np.exp(-50*(xx**2 + yy**2))
f = torch.Tensor(uu)
f = torch.tile(f, (10, 1, 1))

def laplacian(f, stencil):
    f = f.view(f.shape[0], 1, f.shape[1], f.shape[2])
    return F.conv2d(f, stencil.unsqueeze(0).unsqueeze(0), padding=stencil.shape[-1]//2).squeeze()

laplacian_2nd = laplacian(f, laplacian_stencil_2nd)
laplacian_4th = laplacian(f, laplacian_stencil_4th)
laplacian_6th = laplacian(f, laplacian_stencil_6th)
# %%
#3D

laplacian_stencil_3d_2nd = torch.tensor([
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 1, 0], [0, -4, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
], dtype=torch.float32)

import torch

laplacian_stencil_3d_4th = torch.tensor([
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -1/12, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0], [0, 0, 4/3, 0, 0], [0, 4/3, -5/2, 4/3, 0], [0, 0, 4/3, 0, 0], [0, 0, 0, 0, 0]],
    [[0, 0, -1/12, 0, 0], [0, 4/3, -5/2, 4/3, 0], [-1/12, -5/2, 30/4, -5/2, -1/12], [0, 4/3, -5/2, 4/3, 0], [0, 0, -1/12, 0, 0]],
    [[0, 0, 0, 0, 0], [0, 0, 4/3, 0, 0], [0, 4/3, -5/2, 4/3, 0], [0, 0, 4/3, 0, 0], [0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -1/12, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
], dtype=torch.float32)

def laplacian_3d(f, stencil):
    return torch.conv3d(f.unsqueeze(0).unsqueeze(0), stencil.unsqueeze(0).unsqueeze(0), padding=(stencil.shape[0]//2, stencil.shape[1]//2, stencil.shape[2]//2)).squeeze()

# Example usage
f = torch.tile(f, (10, 1, 1))

laplacian_3d_2nd = laplacian_3d(f, laplacian_stencil_3d_2nd)
laplacian_3d_4th = laplacian_3d(f, laplacian_stencil_3d_4th)

# %%
