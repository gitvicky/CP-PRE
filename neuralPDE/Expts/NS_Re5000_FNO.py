#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FNO modelled over the 2D Navier-Stokes equations auto-regressively using the Re5000 dataset from https://zenodo.org/records/7495555

Replicating the results done from the Markov Neural Operator https://arxiv.org/abs/2106.06898

"""

# %%
configuration = {"Case": 'Navier-Stokes',
                 "Field": 'w',
                 "Model": 'FNO',
                 "Epochs": 500,
                 "Batch Size": 50,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GeLU',
                 "Physics Normalisation": 'No',
                 "Normalisation Strategy": 'Min-Max',
                 "T_in": 1,    
                 "T_out": 400,
                 "Step": 1,
                 "Width_time": 64, 
                 "Width_vars": 0,  
                 "Modes": 40,
                 "Variables":1, 
                 "Loss Function": 'LP',
                 "UQ": 'None', #None, Dropout
                 }

# %%
import os
from simvue import Run
run = Run(mode='online')
run.init(folder="/Neural_PDE", tags=['NPDE', 'FNO', 'Tests', 'TW', 'Navier-Stokes', 'Re5000'], metadata=configuration)

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
from Neural_PDE.Models.FNO import *
from Neural_PDE.Utils.processing_utils import * 
from Neural_PDE.Utils.training_utils import * 

# %% 
#Settung up locations. 
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
# Loading Data 
################################################################

t1 = default_timer()
u = np.load(data_loc + '/2D_NS_Re5000.npy')
u = u[:,100:] #Removing the first 100 frames. 
field = 'w'
# %% 
#Time Windowing Function 
def time_windowing(uu):
    bs, nt, nx, ny = uu.shape
    
    # Create an empty list to store tensor pairs
    inputs = []
    targets = []

    # Iterate over the first dimension 'a'
    for i in tqdm(range(bs)):
        # Iterate over the second dimension 'b' up to the second-to-last element
        for j in range(nt - 1):
            # Create a pair of tensors
            inputs.append(uu[i, j:j+1, :, :])
            targets.append(uu[i, j+1: j+2, :, :])

    inputs, targets = np.asarray(inputs), np.asarray(targets)  
    return inputs, targets

# %% 
ntrain = 80
ntest = 20

#Extracting configuration files
T_in = configuration['T_in']
T_out = configuration['T_out']
step = configuration['Step']
modes = configuration['Modes']
width_vars = configuration['Width_vars']
width_time = configuration['Width_time']
output_size = configuration['Step']
num_vars = configuration['Variables']
batch_size = configuration['Batch Size']

#Setting up train and test
train_a, train_u = time_windowing(u[:ntrain])
test_a, test_u = time_windowing(u[ntrain:])

train_a, train_u, test_a, test_u = torch.tensor(train_a).permute(0, 2, 3, 1).unsqueeze(1), torch.tensor(train_u).permute(0, 2, 3, 1).unsqueeze(1), torch.tensor(test_a).permute(0, 2, 3, 1).unsqueeze(1), torch.tensor(test_u).permute(0, 2, 3, 1).unsqueeze(1)

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
elif norm_strategy == 'UnitGaussian':
    normalizer = UnitGaussianNormalizer


a_normalizer = normalizer(train_a)
u_normalizer = normalizer(train_u)

train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

train_u = u_normalizer.encode(train_u)
test_u_encoded = u_normalizer.encode(test_u)

#Saving Normalisation 
saved_normalisations = model_loc + '/' + configuration['Model'] + '_' + configuration['Case'] + '_' + run.name + '_' + 'norms.npz'

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

model = FNO_multi2d(T_in, step, modes, modes, num_vars, width_time)
model.to(device)

run.update_metadata({'Number of Params': int(model.count_params())})
print("Number of model params : " + str(model.count_params()))

#Setting up the optimizer and scheduler, loss and epochs 
optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
loss_func = LpLoss(size_average=False)
epochs = configuration['Epochs']

# %%
####################################
#Training Loop 
####################################
start_time = default_timer()
for ep in range(epochs): #Training Loop - Epochwise

    model.train()
    t1 = default_timer()
    train_loss, test_loss = train_one_epoch(model, train_loader, test_loader, loss_func, optimizer)
    t2 = default_timer()

    train_loss = train_loss / ntrain / num_vars
    test_loss = test_loss / ntest / num_vars

    print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 3)}, Test Loss: {round(test_loss,3)}")
    run.log_metrics({'Train Loss': train_loss, 'Test Loss': test_loss})
    
    scheduler.step()

train_time = default_timer() - start_time


# %%
#Saving the Model
saved_model = model_loc + '/' + configuration['Model'] + '_' + configuration['Case'] + '_' +run.name + '.pth'
torch.save( model.state_dict(), saved_model)
run.save(saved_model, 'output')
# %%
#Validation
pred_set_encoded, mse, mae = validation(model, test_loader)
mse, mae = mse / ntrain , mae / ntest 
# %%
print('(MSE) Testing Error: %.3e' % (mse))
print('(MAE) Testing Error: %.3e' % (mae))

run.update_metadata({'Training Time': float(train_time),
                     'MSE Test Error': float(mse),
                     'MAE Test Error': float(mae)
                    })

#%%
#Denormalising the predictions
pred_set = u_normalizer.decode(pred_set_encoded.to(device)).cpu()
# %% 
test_u = test_u.reshape(ntest, 400, 1, 128, 128)
pred_set = pred_set.reshape(ntest, 400, 1, 128, 128)

#Plotting performance

idx = np.random.randint(0,ntest) 
idx = 0

# %%
u_field = test_u[idx][:, 0] #removing the variable dimension
    
v_min_1 = torch.min(u_field[0])
v_max_1 = torch.max(u_field[0])

v_min_2 = torch.min(u_field[int(T_out/2)])
v_max_2 = torch.max(u_field[int(T_out/2)])

v_min_3 = torch.min(u_field[-1])
v_max_3 = torch.max(u_field[-1])

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(2, 3, 1)
pcm = ax.imshow(u_field[0], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
# ax.title.set_text('Initial')
ax.title.set_text('t=' + str(T_in))
ax.set_ylabel('Solution -  ' + configuration['Field'])
fig.colorbar(pcm, pad=0.05)

ax = fig.add_subplot(2, 3, 2)
pcm = ax.imshow(u_field[int(T_out/2)], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2,
                vmax=v_max_2)
# ax.title.set_text('Middle')
ax.title.set_text('t=' + str(int(T_out / 2)))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)

ax = fig.add_subplot(2, 3, 3)
pcm = ax.imshow(u_field[-1], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
# ax.title.set_text('Final')
ax.title.set_text('t=' + str(T_out))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)

u_field = pred_set[idx][:, 0]

ax = fig.add_subplot(2, 3, 4)
pcm = ax.imshow(u_field[0], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
ax.set_ylabel('FNO' )

fig.colorbar(pcm, pad=0.05)

ax = fig.add_subplot(2, 3, 5)
pcm = ax.imshow(u_field[int(T_out/ 2)], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2,
                vmax=v_max_2)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)

ax = fig.add_subplot(2, 3, 6)
pcm = ax.imshow(u_field[-1], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)

plot_name = plot_loc + '/' + field + '_' + run.name + '.png'
plt.savefig(plot_name)
run.save(plot_name, 'output')

run.close()
# %%
