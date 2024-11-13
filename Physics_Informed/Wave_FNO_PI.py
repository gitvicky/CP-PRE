#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FNO modelled over the 2D Wave Equation auto-regressively - Finetuned without any data in a physics informed manner. 

Equation: u_tt = D*(u_xx + u_yy), D=1.0

"""

# %%
configuration = {"Case": 'Wave',
                 "Field": 'u',
                 "Model": 'FNO',
                 "Epochs": 1,
                 "Batch Size": 1,
                 "Optimizer": 'Adam',
                 "Learning Rate": 1e-2,
                 "Scheduler Step": 500,
                 "Scheduler Gamma": 0.9,
                 "Activation": 'GeLU',
                 "Physics Normalisation": 'No',
                 "Normalisation Strategy": 'Min-Max',
                 "T_in": 20,    
                 "T_out": 40,
                 "Step": 40,
                 "Width_time": 32, 
                 "Width_vars": 0,  
                 "Modes": 16,
                 "Variables":1, 
                 "Loss Function": 'Residual',
                 "UQ": 'None', #None, Dropout
                 "Config": 'basic' #Basic or fine-tune
                 }

# %%
import os
from simvue import Run
run = Run(mode='online')
run.init(folder="/Neural_PDE", tags=['NPDE', 'FNO', 'PRE', 'ConvOps', 'AR', 'Wave', 'Physics-Informed'], metadata=configuration)

#Saving the current run file and the git hash of the repo
run.save_file(os.path.abspath(__file__), 'code')
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
trained_model_loc = os.path.dirname(os.getcwd()) + '/Weights'
model_loc = file_loc + '/Weights'
plot_loc =  os.path.dirname(os.getcwd()) + '/Plots'
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

Nx = 64 # Mesh Discretesiation 
x_min = -1.0 # Minimum value of x
x_max = 1.0 # maximum value of x
y_min = -1.0 # Minimum value of y 
y_max = 1.0 # Minimum value of y
tend = 1
Lambda = 40 #Gaussian Amplitude
aa = 0.25 #X-pos
bb = 0.25 #Y-pos
c = 0.5 #Wave Speed <=1.0

solver = Wave_2D(Nx, x_min, x_max, tend, c)
xx, yy, t, u_sol = solver.solve(Lambda, aa, bb) #solution shape -> t, x, y
u_sol = u_sol[::5]
t = t[::5]

dx = xx[-1] - xx[-2]
dy = yy[-1] - yy[-2]
dt = t[-1] - t[-2]

u = torch.tensor(u_sol, dtype=torch.float32).unsqueeze(0).permute(0, 2, 3, 1).unsqueeze(1)
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


norms = np.load(trained_model_loc + '/FNO_Wave_cyclic-muntin_norms.npz')

a_normalizer = normalizer(train_a)
a_normalizer.a = torch.tensor(norms['in_a'])
a_normalizer.b = torch.tensor(norms['in_b'])

u_normalizer = normalizer(train_u)
u_normalizer.a = torch.tensor(norms['out_a'])
u_normalizer.b = torch.tensor(norms['out_b'])
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
if configuration['Config'] == 'finetune':
    model.load_state_dict(torch.load(trained_model_loc + '/FNO_Wave_cyclic-muntin.pth', map_location='cpu'))
model.to(device)

run.update_metadata({'Number of Params': int(model.count_params())})
print("Number of model params : " + str(model.count_params()))

# %%
#Defining the physics
from Utils.ConvOps_2d import *
dx, dy, dt, c = dx, dy, dt, 0.5
D_tt = ConvOperator(domain ='t', order=2, device=device)#, scale=alpha)
D_xx_yy = ConvOperator(domain=('x','y'), order=2, device=device)#, scale=beta)
D = ConvOperator(device=device) #Additive Kernels
D.kernel = D_tt.kernel - (c*dt/dx)**2 * D_xx_yy.kernel 
D.kernel.requires_grad = True

def residual_loss(field):
    field = field[:, 0, 1:-1, 1:-1, 1:-1].permute(0, 3, 1, 2) #Taking care of the Boundaries. 
    return D(field)

# loss_func = residual_loss
loss_func = LpLoss(size_average=False)

# %%
#Setting up the optimizer and scheduler, loss and epochs 
if configuration['Optimizer'] == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])

if configuration['Optimizer'] == 'LBFGS':
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)
    #Defining the Closure required to evaluate LBFGS
    def closure():
        optimizer.zero_grad()
        y_out = model(xx)
        loss = residual_loss(y_out)
        loss.backward()
        return loss

epochs = configuration['Epochs']

# %% 
#Single Loop
def train_one_epoch_AR():
    model.train()
    # t1 = default_timer()
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        batch_size = xx.shape[0]

        for t in range(0, T_out, step):
            y = yy[..., t:t + step]
            im = model(xx)

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)
            
        #PI Loss
        loss = residual_loss(im).pow(2).mean()

        #LP Loss
        # loss = loss_func(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))

        train_l2_full += loss.item()

        if configuration['Optimizer'] == 'LBFGS':
            optimizer.step(closure)

        elif configuration['Optimizer'] == 'Adam':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    train_loss = train_l2_full 
    # train_l2_step = train_l2_step / ntrain / (T_out / step) /num_vars

    # Validation Loop
    test_loss = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            xx, yy = xx.to(device), yy.to(device)

            for t in range(0, T_out, step):
                y = yy[..., t:t + step]
                out = model(xx)

                if t == 0:
                    pred = out
                else:
                    pred = torch.cat((pred, out), -1)
 
                xx = torch.cat((xx[..., step:], out), dim=-1)
            test_loss += loss_func(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    # t2 = default_timer()

    return train_loss, test_loss #remember to divide the ntrain/ntest and num_vars at the other end before logging.

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
    train_loss, test_loss = train_one_epoch_AR()
    t2 = default_timer()

    train_loss = train_loss / ntrain / num_vars
    test_loss = test_loss / ntest / num_vars

    print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 3)}, Test Loss: {round(test_loss,3)}")
    run.log_metrics({'Train Loss': train_loss, 'Test Loss': test_loss})
    

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

idx=0

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
