# %% 
import numpy as np
import matplotlib.pyplot as plt
import findiff as fd

# %%
x = np.linspace(0,4,1000)
dx = x[-1] - x[-2]
f = np.sin(x)
d_dx = fd.FinDiff(0, dx, 1)
df_dx = d_dx(f)

plt.figure()
plt.plot(x, f,label='func')
plt.plot(x, df_dx, label='deriv')
# %%
#Tring it out on a 2D problem 
x, y = [np.linspace(0, 10, 100)]*2
dx, dy = x[1] - x[0], y[1] - y[0]
X, Y = np.meshgrid(x, y, indexing='ij')
f = np.sin(X) * np.cos(Y)
f = f.T

d_dx = fd.FinDiff(0, dx)
d_dy = fd.FinDiff(1, dy)

df_dx = d_dx(f)
df_dy = d_dy(f)
# %%
#Obtaining FD using ConvOps 
import sys
sys.path.append("..")

from Utils.ConvOps_2d import * 
D_x = ConvOperator(('x'), 1, scale=1/(2*dx))
D_y = ConvOperator(('y'), 1, scale=1/(2*dy))

f_x = D_x(torch.tensor(f, dtype=torch.float32).unsqueeze(0))[1:-1,1:-1] 
f_y = D_y(torch.tensor(f, dtype=torch.float32).unsqueeze(0))[1:-1,1:-1]

plt.figure()
plt.imshow(f_x)
plt.title('ConvDiff')
plt.colorbar()
plt.figure()
plt.imshow(df_dx)
plt.colorbar()
plt.title('FinDiff')


# %%
