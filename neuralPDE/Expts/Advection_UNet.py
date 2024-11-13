"""
Created on 12 Dec 2023

Obtaining the Physics Residuals as a measure of UQ on U-Net surrogate for 1D Advection Equation .

Equation: 
    U_t + v U_x = 0

"""

#%%
#Training Configuration - used as the config file for simvue.
configuration = {"Case": 'Advection',
                 "Field": 'u',
                 "Model": 'UNet',
                 "Epochs": 100,
                 "Batch Size": 50,
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
                 "Variables":1, 
                 "Noise":0.0, 
                 "Loss Function": 'MSE',
                 }

import os
from simvue import Run
run = Run(mode='online')
run.init(folder="/Neural_PDE", tags=['NPDE', 'U-Net'], metadata=configuration)

#Saving the current run file and the git hash of the repo
run.save(os.path.abspath(__file__), 'code')
import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
run.update_metadata({'Git Hash': sha})

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
from Neural_PDE.Models.UNet import *
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
torch.set_default_dtype(torch.float32)

# %% 
#Generating the Datasets by running the simulation
t1 = default_timer()
from Neural_PDE.Numerical_Solvers.Advection.Advection_1D import *
from pyDOE import lhs

#Obtaining the exact and FD solution of the 1D Advection Equation. 

Nx = 200 #Number of x-points
Nt = 50 #Number of time instances 
x_min, x_max = 0.0, 2.0 #X min and max
t_end = 0.5 #time length

sim = Advection_1d(Nx, Nt, x_min, x_max, t_end) 

n_sims = 100

lb = np.asarray([0.1, 0.1]) #pos, velocity
ub = np.asarray([1.0, 1.0])

params = lb + (ub - lb) * lhs(2, n_sims)

u_sol = []
for ii in tqdm(range(n_sims)):
    xc = params[ii, 0]
    v = params[ii, 1]
    x, t, u_soln, u_exact = sim.solve(v, xc)
    u_sol.append(u_soln)

u_sol = np.asarray(u_sol)
u_sol = u_sol[:, :, 1:-2]
x = x[1:-2]
velocity = params[:,1]

# %% 
u = torch.tensor(u_sol, dtype=torch.float32)
u = u.permute(0, 2, 1) #only for FNO
x_grid = x
t_grid = t
# %% 
ntrain = 80
ntest = 20
S = Nx  #Grid Size

#Extracting configuration files
T_in = configuration['T_in']
T_out = configuration['T_out']
step = configuration['Step']
width = configuration['Width']
output_size = configuration['Step']
num_vars = configuration['Variables']
batch_size = configuration['Batch Size']

train_a = u[:ntrain, :, :T_in]
train_u = u[:ntrain, :, T_in:T_out+T_in]

test_a = u[-ntest:, :, :T_in]
test_u = u[-ntest:, :, T_in:T_out+T_in]

print(train_u.shape)
print(test_u.shape)


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
#Setting up the training and testing data splits
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u_encoded), batch_size=batch_size, shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %%
################################################################
# training and evaluation
################################################################
model = UNet1d(in_channels = T_in, out_channels = step, init_features=32)
model.to(device)

run.update_metadata({'Number of Params': int(model.count_params())})
print("Number of model params : " + str(model.count_params()))

#Setting up the optimizer and scheduler, loss and epochs 
optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
loss_func = torch.nn.MSELoss()
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
#Testing 
pred_set_encoded, mse, mae = validation(model, test_a, test_u_encoded, step, T_out)

print('Testing Error (MSE) : %.3e' % (mse))
print('Testing Error (MAE) : %.3e' % (mae))

run.update_metadata({'Training Time': float(train_time),
                     'MSE Test Error': float(mse),
                     'MAE Test Error': float(mae)
                    })

# %% 
#Denormalising the predictions
pred_set = u_normalizer.decode(pred_set_encoded.to(device)).cpu()

# %%
#Plotting the surrogate performance against that of the test data. 

idx = np.random.randint(0,ntest) 
x_range = x_grid

u_field_actual = test_u[idx]
u_field_pred = pred_set[idx]

v_min = torch.min(u_field_actual)
v_max = torch.max(u_field_actual)


fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1,3,1)
pcm = ax.plot(x_range, u_field_actual[0,:], color='green')
pcm = ax.plot(x_range, u_field_pred[0,:], color='firebrick')
ax.set_ylim([v_min, v_max])
ax.title.set_text('t='+ str(T_in))

u_field_actual = test_u[idx]
u_field_pred = pred_set[idx]

ax = fig.add_subplot(1,3,2)
pcm = ax.plot(x_range, u_field_actual[int(T/2),:], color='green')
pcm = ax.plot(x_range, u_field_pred[int(T/2),:], color='firebrick')
ax.set_ylim([v_min, v_max])
ax.title.set_text('t='+ str(int((T+(T_in/2)))))
ax.axes.yaxis.set_ticks([])

ax = fig.add_subplot(1,3,3)
pcm = ax.plot(x_range, u_field_actual[-1,:], color='green')
pcm = ax.plot(x_range, u_field_pred[-1,:], color='firebrick')
ax.title.set_text('t='+str(T+T_in))
ax.set_ylim([v_min, v_max])
ax.axes.yaxis.set_ticks([])

plot_name = plot_loc + '/' + configuration['Field'] + '_' + run.name + '.png'
plt.savefig(plot_name)
run.save(plot_name, 'output')


run.close()
