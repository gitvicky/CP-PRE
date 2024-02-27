#Testing out the stencil inversion methods

# %%
import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torch.nn.functional as F

# %% 
#Generating arbitraty data over which we will perform the convolutions. 
#IC of the Wave case
def Wave_IC(Lambda=-10, x_pos = 0.25, y_pos=0.25):
    N = 30 # Mesh Discretesiation 
    x0 = -1.0 # Minimum value of x
    xf = 1.0 # maximum value of x
    y0 = -1.0 # Minimum value of y 
    yf = 1.0 # Minimum value of y

    k = np.arange(N + 1)
    x = np.cos(k*np.pi/N) #Creating the x and y discretisations
    y = x.copy()
    xx, yy = np.meshgrid(x, y)


    #Initial Conditions 
    return np.exp(Lambda*((xx-x_pos)**2 + (yy-y_pos)**2))
  
uu = Wave_IC()
uu = torch.tensor(uu, dtype=torch.float32)
# %%
#Arbitrary stencil -- created with random numbers 
kernel = torch.tensor([[-4., 3., -1.],
                           [3., 8., -2.],
                           [4., -1., 5.]])

if torch.linalg.det(kernel) != 0:
    kernel_inv = torch.linalg.inv(kernel)
kernel_transp = kernel.T
# %%
#Testing out to see if the inverse of the kernel 
#used for transposed convolution could retreieve the original. 

conv = F.conv2d(uu.view(1,1,uu.shape[0], uu.shape[1]), kernel.view(1,1,3,3,))
inv_conv = F.conv_transpose2d(conv, kernel_inv.view(1,1,3,3,))
transposed_conv = F.conv_transpose2d(conv, kernel_transp.view(1,1,3,3,))

# %%
#Plotting the conv and inv_conv
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm 

fig = plt.figure(figsize=(20, 5))
# plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.1)
plt.title('Arbitrary stencil')
ax = fig.add_subplot(1,3,1)
pcm =ax.imshow(uu, cmap=cm.coolwarm)
ax.title.set_text('Actual')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(1,3,2)
pcm =ax.imshow(inv_conv[0,0], cmap=cm.coolwarm)
ax.title.set_text('Retrieved using inverse')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(1,3,3)
pcm =ax.imshow(transposed_conv[0,0, 2:-2, 2:-2], cmap=cm.coolwarm)
ax.title.set_text('Retrieved using transpose')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))



# %%
#FD Stencil 
kernel = torch.tensor([[0., 1., 0.],
                           [1., -4., 1.],
                           [0, 1., 0.]])
kernel_transp = kernel.T
# %%
#Testing out to see if the inverse of the kernel 
#used for transposed convolution could retreieve the original. 

conv = F.conv2d(uu.view(1,1,uu.shape[0], uu.shape[1]), kernel.view(1,1,3,3,))
transposed_conv = F.conv_transpose2d(conv, kernel_transp.view(1,1,3,3,))

# %%
#Plotting the conv and inv_conv
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm 

fig = plt.figure(figsize=(10, 5))
# plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.1)
plt.title('FD stencil')
ax = fig.add_subplot(1,2,1)
pcm =ax.imshow(uu, cmap=cm.coolwarm)
ax.title.set_text('Actual')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(1,2,2)
pcm =ax.imshow(transposed_conv[0,0, 2:-2, 2:-2], cmap=cm.coolwarm)
ax.title.set_text('Retrieved using Transpose')
ax.set_xlabel('x')
ax.set_ylabel('y')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.formatter.set_powerlimits((0, 0))

# %%
