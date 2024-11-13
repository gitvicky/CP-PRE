#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FNO modelled over the 2D Wave Equation auto-regressively - Finetuned without any data in a physics informed manner. 

Equation: u_tt = D*(u_xx + u_yy), D=1.0

"""

# %%
configuration = {"Case": 'Advection',
                 "Field": 'u',
                 "Model": 'FNO',
                 "Epochs": 500,
                 "Batch Size": 100,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.001,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'Tanh',
                 "Normalisation Strategy": 'Identity',
                 "T_in": 20,    
                 "T_out": 30,
                 "Step": 30,
                 "Width": 32, 
                 "Modes": 8,
                 "Variables":1,
                 "Loss Function": 'Residual',
                 "UQ": 'None', #None, Dropout
                 "Config": 'basic', #Basic or fine-tune
                 "n_train": 120
                 }

# %%
import os
from simvue import Run
run = Run(mode='online')
run.init(folder="/Neural_PDE", tags=['NPDE', 'FNO', 'Tests', 'Advection', 'Physics-Informed'], metadata=configuration)

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
import sys
sys.path.append("..")

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
t1 = default_timer()

from Neural_PDE.Numerical_Solvers.Advection.Advection_1D import *
from pyDOE import lhs

Nx = 200 #Number of x-points
Nt = 50 #Number of time instances 
x_min, x_max = 0.0, 2.0 #X min and max
t_end = 0.5 #time length

sim = Advection_1d(Nx, Nt, x_min, x_max, t_end) 

n_train = configuration['n_train']

lb = np.asarray([0.1, 0.1]) #pos, velocity
ub = np.asarray([0.7, 0.7])

params = lb + (ub - lb) * lhs(2, n_train)

u_sol = []
for ii in tqdm(range(n_train)):
    xc = params[ii, 0]
    v = params[ii, 1]
    x, t, u_soln, u_exact = sim.solve(v, xc)
    u_sol.append(u_soln)

u_sol = np.asarray(u_sol)
u_sol = u_sol[:, :, 1:-2]
x = x[1:-2]
velocity = params[:,1]

u = torch.tensor(u_sol, dtype=torch.float32).permute(0, 2, 1).unsqueeze(1)
# %% 
ntrain = 100
ntest = 20
S = 200 #Grid Size

#Extracting configuration files
T_in = configuration['T_in']
T_out = configuration['T_out']
step = configuration['Step']
width = configuration['Width']
modes = configuration['Modes']
output_size = configuration['Step']
num_vars = configuration['Variables']
batch_size = configuration['Batch Size']

#Setting up train and test
train_a = u[:ntrain,:,:,:T_in]
train_u = u[:ntrain,:,:,T_in:T_out+T_in]

test_a = u[-ntest:,:,:,:T_in]
test_u = u[-ntest:,:,:,T_in:T_out+T_in]

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
# %%
# #Saving Normalisation 
# saved_normalisations = model_loc + '/' + configuration['Model'] + '_' + configuration['Case'] + '_' + run.name + '_' + 'norms.npz'

# np.savez(saved_normalisations, 
#         in_a=a_normalizer.a.numpy(), in_b=a_normalizer.b.numpy(), 
#         out_a=u_normalizer.a.numpy(), out_b=u_normalizer.b.numpy()
#         )

# run.save(saved_normalisations, 'output')
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

model = FNO_multi1d(T_in, step, modes, num_vars, width, width_vars=0)
# if configuration['Config'] == 'finetune':
#     model.load_state_dict(torch.load(model_loc + '/FNO_Wave_charitable-sea.pth', map_location='cpu'))
model.to(device)

run.update_metadata({'Number of Params': int(model.count_params())})
print("Number of model params : " + str(model.count_params()))

#Setting up the optimizer and scheduler, loss and epochs 
optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
epochs = configuration['Epochs']

#Defining the physics
from Utils.ConvOps_1d import *
dx = torch.tensor(x[-1] - x[-2], dtype=torch.float32)
dt = torch.tensor(t[-1] - t[-2], dtype=torch.float32)
v = torch.tensor(velocity[:ntrain], dtype=torch.float32)

D_t = ConvOperator(domain='t', order=1)#, scale=alpha)
D_x = ConvOperator(domain='x', order=1)#, scale=beta) 
# D = ConvOperator(device=device) #Additive Kernels
# D.kernel = D_t.kernel + (v*dt/dx) * D_x.kernel

def residual_loss(field):
    field = field[:, 0, 1:-1, 1:-1].permute(0, 1, 2) #Taking care of the Boundaries. 
    res =  D_t(field) + (v*dt/dx) * D_x(field)
    return res

loss_func = residual_loss
    
# %%
####################################
#Training Loop - Residual Losses
####################################
if device == 'cuda':
    u_normalizer.cuda()

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


# start_time = default_timer()
# for ep in range(epochs): #Training Loop - Epochwise
#     model.train()
#     t1 = default_timer()
#     train_loss = 0
#     test_loss = 0 
#     for xx, yy in train_loader:
#         optimizer.zero_grad()
#         xx = xx.to(device)
#         yy = yy.to(device)

#         pred = model(xx)
#         pred = u_normalizer.decode(pred)
#         loss = residual_loss(pred).pow(2).mean()
#         # loss = (pred-yy).pow(2).mean()
#         train_loss += loss

#     train_loss.backward()
#     optimizer.step()
    
#     #Validation - L2 Error 
#     with torch.no_grad():
#         for xx, yy in test_loader:
#             xx = xx.to(device)
#             yy = yy.to(device)
#             batch_size = xx.shape[0]
            
#             pred = model(xx)
#             loss = (pred - yy).pow(2).mean()
#             test_loss += loss

#     t2 = default_timer()

#     train_loss = train_loss.item()
#     test_loss = test_loss.item()

#     print(f"Epoch {ep}, Time Taken: {round(t2-t1, 6)}, Train Loss: {round(train_loss, 6)}, Test Loss: {round(test_loss, 6)}")
#     run.log_metrics({'Train Loss': train_loss, 'Test Loss': test_loss})
    
#     scheduler.step()

# train_time = default_timer() - start_time

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

idx=0

u_field_soln = test_u[idx]
u_field_pred = pred_set[idx]

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 3, 1)
ax.plot(x, u_field_soln[0, :, 0], label='Soln.')
ax.plot(x, u_field_pred[0, :, 0], label='Pred.')
ax.title.set_text('t=' + str(T_in))
ax.set_xlabel('x')
ax.set_ylabel('u')

ax = fig.add_subplot(1, 3, 2)
ax.plot(x, u_field_soln[0, :, int(T_out/ 2)], label='Soln.')
ax.plot(x, u_field_pred[0, :, int(T_out/ 2)], label='Pred.')
ax.title.set_text('t=' + str(int(T_in + (T_out / 2))))
ax.set_xlabel('x')
# ax.set_ylabel('u')

ax = fig.add_subplot(1, 3, 3)
ax.plot(x, u_field_soln[0, :, -1], label='Soln.')
ax.plot(x, u_field_pred[0, :, -1], label='Pred.')
ax.title.set_text('t=' + str(T_out+ T_in))
ax.set_xlabel('x')
# ax.set_ylabel('u')
plt.legend()

plot_name = plot_loc + '/' + configuration['Field'] + '_' + run.name + '.png'
plt.savefig(plot_name)
run.save(plot_name, 'output')

run.close()
# %%
