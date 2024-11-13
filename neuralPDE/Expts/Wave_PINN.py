#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Siren modelled over the 2D Wave Equation as a PINN.- rewrite

Equation: u_tt = D*(u_xx + u_yy), D=1.0

"""

# %%
configuration = {"Case": 'Wave',
                 "Field": 'u',
                 "Model": 'Siren',
                 "Epochs": 1000,
                 "Batch Size": 50000,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GeLU',
                 "Physics Normalisation": 'No',
                 "Normalisation Strategy": 'Identity',
                 "T_range": 80, #Full range of time instances
                 "Layers": 5,                 
                 "Width": 32, 
                 "Coords": 3, #Number of spatio-temporal coordinates - would form the number of inputs 
                 "Variables":1, #Number of variables being modelled - would form the number of outputs. 
                 "Points":int(1e6), #Points sampled for training. 10 percent of that is used for testing. 
                 "Loss Function": 'MSE',
                 "UQ": 'None', #None, Dropout
                 }

# %%
import os
from simvue import Run
run = Run(mode='online')
run.init(folder="/Neural_PDE", tags=['NPDE', 'Siren', 'Tests', 'INR', 'PINN'], metadata=configuration)

#Saving the current run file and the git hash of the repo
run.save(os.path.abspath(__file__), 'code')
import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
run.update_metadata({'Git Hash': sha})

# %% 
#Importing the necessary packages
import sys
import numpy as np
from tqdm import tqdm 
import torch
import matplotlib
import matplotlib.pyplot as plt
import time 
from timeit import default_timer
from tqdm import tqdm 

#Adding the NPDE package to the system python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
# %%
#Importing the models and utilities. 
from Neural_PDE.Models.INR import *
from Neural_PDE.Utils.processing_utils import * 
from Neural_PDE.Utils.training_utils import * 

# %% 
#Setting up locations. 
file_loc = os.getcwd()
data_loc = os.path.dirname(os.getcwd()) + '/Data'
model_loc = file_loc + '/Weights'
plot_loc = file_loc + '/Plots'
#Setting up the seeds and devices
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
################################################################
# Loading (Generating) Data 
################################################################

t1 = default_timer()
from Neural_PDE.Numerical_Solvers.Wave.Wave_2D_Spectral import * 

#Example of Usage
Nx = 32 # Mesh Discretesiation 
Nt = 100 #Max time
x_min = -1.0 # Minimum value of x
x_max = 1.0 # maximum value of x
y_min = -1.0 # Minimum value of y 
y_max = 1.0 # Minimum value of y
tend = 1
Lambda = 20
aa = 0.25
bb = 0.25
c = 1.0 # Wave Speed <=1.0


#Initialising the Solver
solver = Wave_2D(Nx, x_min, x_max, tend, c)
x, y, t, u_sol = solver.solve(Lambda, aa, bb)

u = torch.tensor(u_sol, dtype=torch.float32)
u = u.permute(1,2,0)

xx, yy, tt = np.meshgrid(x, y, t)

#Stacked cooredinate values and corresponding field values stacked. 
aa = np.vstack((xx.flatten(), yy.flatten(), tt.flatten() )).T
uu = u.flatten().unsqueeze(-1)

#coordinates 
coords = torch.tensor(aa, dtype=torch.float32)

# %% 
ntrain = 1 #Only 1 simulation is used. 
ntest = 1 #Only 1 simulation is used. 
S = Nx #Grid Size

#Extracting configuration files
num_coords = configuration['Coords']
num_vars = configuration['Variables']
layers = configuration['Layers']
width = configuration['Width']
batch_size = configuration['Batch Size']
T_range = configuration['T_range']
num_points = configuration["Points"]

#Slicing the fields and setting up the coordinate meshes. 
t = t[:T_range]
u = u[...,:T_range]
xx, yy, tt = np.meshgrid(x, y, t)

aa = np.vstack((xx.flatten(), yy.flatten(), tt.flatten() )).T
uu = u.flatten().unsqueeze(dim=-1)
# %%
#Setting up train and test
tts = 0.8 #test train split
ntrain = int(tts * len(aa)) 
ntest = len(aa) - ntrain
#Splitting based on the simulations

train_index = np.random.choice(len(aa), ntrain, replace=False)
mask = np.ones(len(aa), dtype=bool)
mask[train_index.tolist()] = False
test_index = mask

train_a = torch.tensor(aa, dtype=torch.float32)[train_index]
train_u = uu[train_index]

test_a = torch.tensor(aa, dtype=torch.float32)[test_index]
test_u = uu[test_index]

print("Training Input: " + str(train_a.shape))
print("Training Output: " + str(train_u.shape))

# %%
#Normalising the train and test datasets with the preferred normalisation. 

norm_strategy = configuration['Normalisation Strategy']

if norm_strategy == 'Min-Max':
    normalizer = MinMax_Normalizer
elif norm_strategy == 'Range':
    normalizer = RangeNormalizer
elif norm_strategy == 'Gaussian':
    normalizer = GaussianNormalizer
elif norm_strategy == 'Identity':
    normalizer = Identity

a_normalizer = normalizer(train_a)
u_normalizer = normalizer(train_u)

train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

train_u = u_normalizer.encode(train_u)
test_u_encoded = u_normalizer.encode(test_u)

#Saving Normalisation 
saved_normalisations = model_loc + '/' + configuration['Model'] + '_' + configuration['Case'] + '_' +run.name + '_' + 'norms.npz'

np.savez(saved_normalisations, 
        in_a=a_normalizer.a.numpy(), in_b=a_normalizer.b.numpy(), 
        out_a=u_normalizer.a.numpy(), out_b=u_normalizer.b.numpy()
        )

run.save(saved_normalisations, 'output')
# %%
#Setting up the data loaders. 
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u_encoded), batch_size=batch_size, shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %%
################################################################
# training and evaluation
################################################################

model = Siren(in_features=num_coords, hidden_features=width, hidden_layers=layers, out_features=num_vars)
model.to(device)

run.update_metadata({'Number of Params': int(model.count_params())})
print("Number of model params : " + str(model.count_params()))

#Setting up the optimizer and scheduler, loss and epochs 
optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
epochs = configuration['Epochs']

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def loss_func(model, xx):
    outs, coords = model(xx)
    derivs = gradient(gradient(outs, coords), coords)
    pde = derivs[2] - (derivs[0] + derivs[1])
    return pde.pow(2).mean()

# %%
####################################
#Training Loop 
####################################
start_time = default_timer()
model.train()
for ep in range(epochs): #Training Loop - Epochwise
    optimizer.zero_grad()
    t1 = default_timer()
    xx, yy = train_a, train_u
    xx, yy = xx.to(device), yy.to(device)
    loss = loss_func(model, xx)
    # train_loss, test_loss = train_one_epoch_PINN(model, train_loader, test_loader, loss_func, optimizer)
    t2 = default_timer()

    loss.backward()
    optimizer.step()
    train_loss = loss.item()#train_loss / ntrain
    test_loss = 0#test_loss / ntest

    print(f"Epoch {ep}, Time Taken: {round(t2-t1,2)}, Train Loss: {round(train_loss, 5)}, Test Loss: {round(test_loss,5)}")
    run.log_metrics({'Train Loss': train_loss, 'Test Loss': test_loss})
    
    scheduler.step()

train_time = default_timer() - start_time


# %%
#Saving the Model
saved_model = model_loc + '/' + configuration['Model'] + '_' + configuration['Case'] + '_' +run.name + '.pth'
torch.save( model.state_dict(), saved_model)
run.save(saved_model, 'output')
# %%
#Validation using the data split simulation-wise. 
test_a = torch.tensor(aa, dtype=torch.float32)
test_a = a_normalizer.encode(test_a)
test_u = uu
test_u_encoded = u_normalizer.encode(test_u)
pred_set_encoded, mse, mae = validation(model, test_a, test_u_encoded)
# %%
print('(MSE) Testing Error: %.3e' % (mse))
print('(MAE) Testing Error: %.3e' % (mae))

run.update_metadata({'Training Time': float(train_time),
                     'MSE Test Error': float(mse),
                     'MAE Test Error': float(mae)
                    })

#%%
#Denormalising the predictions
pred_set = u_normalizer.decode(pred_set_encoded.to(device)).cpu().detach().numpy()

# Rearranging the Predictions for Evaluation. 
test_u = test_u.reshape(1,num_vars,S,S,T_range)
pred_set = pred_set.reshape(1,num_vars, S, S, T_range)
# %% 
#Plotting performance
idx=0
u_field = test_u[idx]
    
v_min_1 = torch.min(u_field[0, :, :, 0])
v_max_1 = torch.max(u_field[0, :, :, 0])

v_min_2 = torch.min(u_field[0, :, :, int(T_range/ 2)])
v_max_2 = torch.max(u_field[0, :, :, int(T_range/ 2)])

v_min_3 = torch.min(u_field[0, :, :, -1])
v_max_3 = torch.max(u_field[0, :, :, -1])

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(2, 3, 1)
pcm = ax.imshow(u_field[0, :, :, 0], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
# ax.title.set_text('Initial')
ax.title.set_text('t=' + str('0'))
ax.set_ylabel('Solution')
fig.colorbar(pcm, pad=0.05)

ax = fig.add_subplot(2, 3, 2)
pcm = ax.imshow(u_field[0, :, :, int(T_range/ 2)], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2,
                vmax=v_max_2)
# ax.title.set_text('Middle')
ax.title.set_text('t=' + str(int((T_range) / 2)))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)

ax = fig.add_subplot(2, 3, 3)
pcm = ax.imshow(u_field[0, :, :, -1], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
# ax.title.set_text('Final')
ax.title.set_text('t=' + str(T_range))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)

u_field = pred_set[idx]

ax = fig.add_subplot(2, 3, 4)
pcm = ax.imshow(u_field[0, :, :, 0], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
ax.set_ylabel('SirenNet')

fig.colorbar(pcm, pad=0.05)

ax = fig.add_subplot(2, 3, 5)
pcm = ax.imshow(u_field[0, :, :, int(T_range/ 2)], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2,
                vmax=v_max_2)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)

ax = fig.add_subplot(2, 3, 6)
pcm = ax.imshow(u_field[0, :, :, -1], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)


plot_name = plot_loc + '/' + configuration['Field'] + '_' + run.name + '.png'
plt.savefig(plot_name)
run.save(plot_name, 'output')

run.close()
# %%
