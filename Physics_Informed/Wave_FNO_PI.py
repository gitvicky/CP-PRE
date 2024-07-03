#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FNO modelled over the 2D Wave Equation auto-regressively - Trained without any data in a physics informed manner. 

Equation: u_tt = D*(u_xx + u_yy), D=1.0

"""

# %%
configuration = {"Case": 'Wave',
                 "Field": 'u',
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
                 "T_in": 20,    
                 "T_out": 60,
                 "Step": 10,
                 "Width_time": 32, 
                 "Width_vars": 0,  
                 "Modes": 8,
                 "Variables":1, 
                 "Loss Function": 'Residual',
                 "UQ": 'None', #None, Dropout
                 "Config": 'Basic' #Basic or fine-tune
                 }

# %%
import os
from simvue import Run
run = Run(mode='online')
run.init(folder="/Neural_PDE", tags=['NPDE', 'FNO', 'Tests', 'AR'], metadata=configuration)

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
# Generating Numerical Simulation
################################################################
from Neural_PDE.Numerical_Solvers.Wave.Wave_2D_Spectral import * 

t1 = default_timer()

Lambda = 20 #Amplitude of the initial Gaussian. 
a = 0.25 #x-position of initial gaussian
b = -0.25 #y-position of initial gaussian 

solver = Wave_2D(Lambda, a , b)
xx, yy, t, u_sol = solver.solve() #solution shape -> t, x, y

dx = xx[-1] - xx[-2]
dy = yy[-1] - yy[-2]
dt = t[-1] - t[-2]

u_sol = torch.tensor(u_sol, dtype=torch.float32).unsqueeze(0).permute(0, 2, 3, 1)
u =u_sol
# %% 
ntrain = 1
ntest = 1
S = 64 #Grid Size

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
train_a = u[:ntrain,:,:,:,:T_in]
train_u = u[:ntrain,:,:,:,T_in:T_out+T_in]

test_a = u[-ntest:,:,:,:,:T_in]
test_u = u[-ntest:,:,:,:,T_in:T_out+T_in]

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

a_normalizer = normalizer(train_a)
u_normalizer = normalizer(train_u)

# %% 
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

train_u = u_normalizer.encode(train_u)
test_u_encoded = u_normalizer.encode(test_u)
# %%
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
# model.load_state_dict(torch.load(model_loc + '/FNO_Wave_null-shape.pth', map_location='cpu'))
model.to(device)

run.update_metadata({'Number of Params': int(model.count_params())})
print("Number of model params : " + str(model.count_params()))

#Setting up the optimizer and scheduler, loss and epochs 
optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
epochs = configuration['Epochs']

#Defining the physics
from Utils.ConvOps_2d import *
dx, dy, dt, c = dx, dy, dt, 1
D_tt = ConvOperator('t', 2)#, scale=alpha)
D_xx_yy = ConvOperator(('x','y'), 2)#, scale=beta)
D = ConvOperator() #Additive Kernels
D.kernel = D_tt.kernel - (c*dt/dx)**2 * D_xx_yy.kernel 

def residual_loss(field):
    return D(field)

loss_func = residual_loss
    
# %%
####################################
#Training Loop 
####################################
start_time = default_timer()
for ep in range(epochs): #Training Loop - Epochwise

    model.train()
    t1 = default_timer()
    train_loss, test_loss = train_one_epoch_AR(model, train_loader, test_loader, loss_func, optimizer, step, T_out)
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
pred_set_encoded, mse, mae = validation_AR(model, test_a, test_u_encoded, step, T_out)
# %%
print('Testing Error (MSE) : %.3e' % (mse))
print('Testing Error (MAE) : %.3e' % (mae))

run.update_metadata({'Training Time': float(train_time),
                     'MSE Test Error': float(mse),
                     'MAE Test Error': float(mae)
                    })

#%%
#Denormalising the predictions
pred_set = u_normalizer.decode(pred_set_encoded.to(device)).cpu()

# %% 
#Plotting performance

idx = np.random.randint(0,ntest) 
idx = 5

# %%

u_field = test_u[idx]
    
v_min_1 = torch.min(u_field[0, :, :, 0])
v_max_1 = torch.max(u_field[0, :, :, 0])

v_min_2 = torch.min(u_field[0, :, :, int(T_out/ 2)])
v_max_2 = torch.max(u_field[0, :, :, int(T_out/ 2)])

v_min_3 = torch.min(u_field[0, :, :, -1])
v_max_3 = torch.max(u_field[0, :, :, -1])

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(2, 3, 1)
pcm = ax.imshow(u_field[0, :, :, 0], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
# ax.title.set_text('Initial')
ax.title.set_text('t=' + str(T_in))
ax.set_ylabel('Solution')
fig.colorbar(pcm, pad=0.05)

ax = fig.add_subplot(2, 3, 2)
pcm = ax.imshow(u_field[0, :, :, int(T_out/ 2)], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2,
                vmax=v_max_2)
# ax.title.set_text('Middle')
ax.title.set_text('t=' + str(int((T_out+ T_in) / 2)))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)

ax = fig.add_subplot(2, 3, 3)
pcm = ax.imshow(u_field[0, :, :, -1], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
# ax.title.set_text('Final')
ax.title.set_text('t=' + str(T_out+ T_in))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)

u_field = pred_set[idx]

ax = fig.add_subplot(2, 3, 4)
pcm = ax.imshow(u_field[0, :, :, 0], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
ax.set_ylabel('FNO')

fig.colorbar(pcm, pad=0.05)

ax = fig.add_subplot(2, 3, 5)
pcm = ax.imshow(u_field[0, :, :, int(T_out/ 2)], cmap=matplotlib.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2,
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
