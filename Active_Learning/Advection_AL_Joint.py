#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uncertainty quantification of a Neural PDE Surrogate solving the Advection Equation using Physics Residuals and guaranteed using Conformal Prediction
Prediction Selection/Rejection based on CP bounds. 

Equation :  U_t + v U_x = 0
Surrogate Model : FNO
Numerical Solver : https://github.com/gitvicky/Neural_PDE/blob/main/Numerical_Solvers/Advection/Advection_1D.py
"""

# %% Training Configuration - used as the config file for simvue.
configuration = {"Case": 'Advection',
                 "Field": 'u',
                 "Model": 'FNO',
                 "Epochs": 100,
                 "Batch Size": 100,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.001,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'Tanh',
                 "Normalisation Strategy": 'Identity',
                 "T_in": 1,    
                 "T_out": 10,
                 "Step": 1,
                 "Width": 8, 
                 "Modes": 4,
                 "Variables":1, 
                 "Noise":0.0, 
                 "Loss Function": 'MSE',
                 "n_train": 100,
                 "n_test": 1000,
                 "n_cal": 1000,
                 "n_pred": 100
                 }


import os
from simvue import Run
run = Run(mode='online')
run.init(folder="/Neural_PDE/Active_Learning", tags=['NPDE', 'FNO', 'PIUQ', 'AR', 'Wave', 'AL'], metadata=configuration)

#Saving the current run file and the git hash of the repo
run.save_file(os.path.abspath(__file__), 'code')
import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
run.update_metadata({'Git Hash': sha})

# %% 
#Importing the necessary packages
import copy
import sys
import numpy as np
from tqdm import tqdm 
import torch
import matplotlib
import matplotlib.pyplot as plt
import time 
from timeit import default_timer
from tqdm import tqdm 

# %%
#Importing the models and utilities. 
import sys
sys.path.append("..")
from Neural_PDE.Models.FNO import *
from Neural_PDE.Utils.processing_utils import * 
from Neural_PDE.Utils.training_utils import * 
from Neural_PDE.UQ.inductive_cp import * 

#Setting up locations. 
file_loc = os.getcwd()
model_loc = file_loc + '/Weights'
plot_loc = file_loc + '/Plots'
data_loc = file_loc + '/Data'

#Setting up the seeds and devices
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

# %% 
#Extracting configuration files
T_in = configuration['T_in']
T_out = configuration['T_out']
step = configuration['Step']
width = configuration['Width']
modes = configuration['Modes']
output_size = configuration['Step']
num_vars = configuration['Variables']
batch_size = configuration['Batch Size']

# %% 
#Simulation Setup
from Neural_PDE.Numerical_Solvers.Advection.Advection_1D import *
from pyDOE import lhs

#Obtaining the exact and FD solution of the 1D Advection Equation. 
Nx = 200 #Number of x-points
Nt = 50 #Number of time instances 
x_min, x_max = 0.0, 2.0 #X min and max
t_end = 0.5 #time length
v = 1.0
sim = Advection_1d(Nx, Nt, x_min, x_max, t_end) 
dt, dx = sim.dt, sim.dx
# %% 
#Utility Functions

def gen_data(params):
    #Generating Data 
    u_sol = []
    for ii in tqdm(range(len(params))):
        xc = params[ii, 0]
        amp = params[ii, 1]
        x, t, u_soln, u_exact = sim.solve(xc, amp, v)
        u_sol.append(u_soln)

    #Extraction
    u_sol = np.asarray(u_sol)
    u_sol = u_sol[:, :, 1:-2]
    x = x[1:-2]

    #Tensorize
    u = torch.tensor(u_sol, dtype=torch.float32)
    u = u.permute(0, 2, 1) #only for FNO
    u = u.unsqueeze(1)

    return x, t, u

#Generate Initial Conditions
def gen_ic(params):
    u_ic = []
    for ii in tqdm(range(len(params))):
        xc = params[ii, 0]
        amp = params[ii, 1]
        sim.initializeU(xc, amp)
        u_ic.append(sim.u)

    u_ic = np.asarray(u_ic)[:, 1:-2]

    u_ic = torch.tensor(u_ic, dtype=torch.float32).unsqueeze(1).unsqueeze(-1)
    return u_ic

#Load Simulation data into Dataloader
def data_loader(uu, dataloader=True, shuffle=True):

    a = uu[:, :, :, :T_in]
    u = uu[:, :, :, T_in:T_out+T_in]

    # print("Input: " + str(a.shape))
    # print("Output: " + str(u.shape))

    #No Normalisation -- Normalisation = Identity 

    if dataloader:
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(a, u), batch_size=batch_size, shuffle=shuffle)
    else:
        loader = [a,u]

    return loader


def train(epochs, model, train_loader, test_loader, loss_func, optimizer, scheduler):
    ####################################
    #Training Loop 
    ####################################
    for ep in range(epochs): #Training Loop - Epochwise

        model.train()
        t1 = default_timer()
        train_loss, test_loss = train_one_epoch_AR(model, train_loader, test_loader, loss_func, optimizer, step, T_out)
        t2 = default_timer()

        # train_loss = train_loss / ntrain / num_vars
        # test_loss = test_loss / ntest / num_vars

        print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 3)}, Test Loss: {round(test_loss,3)}")
        run.log_metrics({'Train Loss': train_loss, 'Test Loss': test_loss})
  
        scheduler.step()
    return model 
# %% 
#Define Bounds
lb = np.asarray([0.5, 50]) #pos, amplitude
ub = np.asarray([1.0, 200])

#Conv kernels --> Stencils 
from Utils.ConvOps_1d import ConvOperator
#Defining the required Convolutional Operations. 
D_t = ConvOperator(domain='t', order=1)#, scale=alpha)
D_x = ConvOperator(domain='x', order=1)#, scale=beta) 

#Residual - Additive Kernels
D = ConvOperator()
D.kernel = D_t.kernel + (v*dt/dx) * D_x.kernel

################################################################
#Train Data
params = lb + (ub - lb) * lhs(2, configuration['n_train'])
x, t, u_sol_train = gen_data(params)
train_loader = data_loader(u_sol_train)
test_loader = data_loader(u_sol_train[-10:]) #Just kept to hefty evaluations each epoch. 

#Test Data
params = lb + (ub - lb) * lhs(2, configuration['n_test'])
x, t, u_sol = gen_data(params)
test_a, test_u = data_loader(u_sol, dataloader=False, shuffle=False)
test_mse = []

#Initialising the model
model = FNO_multi1d(T_in, step, modes, num_vars, width, width_vars=0)
model.to(device)
print("Number of model params : " + str(model.count_params()))
run.update_metadata({'Number of Params': int(model.count_params())})

#Setting up the optimizer and scheduler, loss and epochs 
optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
loss_func = torch.nn.MSELoss()
epochs = configuration['Epochs']

#Training
start_time = default_timer()
model = train(epochs, model, train_loader, test_loader, loss_func, optimizer, scheduler)
train_time = default_timer() - start_time
trained_model = copy.deepcopy(model)


#Evaluation
pred_test, mse, mae = validation_AR(model, test_a, test_u, step, T_out)
test_mse.append(mse)
print('Testing Error (MSE) : %.3e' % (mse))
print('Testing Error (MAE) : %.3e' % (mae))
print()

run.update_metadata({'Initial Train Time': float(train_time),
                     'MSE Test Error': float(mse),
                     'MAE Test Error': float(mae)
                    })

# %% 

## Using Calibration Data from smaller sample of simulations
params = lb + (ub - lb) * lhs(2, configuration['n_cal'])
x, t, u_sol = gen_data(params)
u_in_cal, u_out_cal = data_loader(u_sol, dataloader=False, shuffle=False)
u_pred_cal, mse, mae = validation_AR(model, u_in_cal, u_out_cal, step, T_out)

residual_out_cal = D(u_out_cal.permute(0,1,3,2)[:,0])[...,1:-1, 1:-1]
residual_pred_cal = D(u_pred_cal.permute(0,1,3,2)[:,0])[...,1:-1, 1:-1]

# %%
#Prediction Residuals 
params = lb + (ub - lb) * lhs(2, configuration['n_pred'])
u_in_pred = gen_ic(params)
pred_pred, mse, mae = validation_AR(model, u_in_pred, torch.zeros((u_in_pred.shape[0], u_in_pred.shape[1], u_in_pred.shape[2], T_out)), configuration['Step'], configuration['T_out'])
pred_pred = pred_pred.permute(0,1,3,2)[:,0]
uu_pred = pred_pred
pred_residual = D(uu_pred)[...,1:-1, 1:-1]

# res = residual_out_cal #Data-Driven
res = residual_pred_cal #Physics-Driven

modulation = modulation_func(res.numpy(), np.zeros(res.shape))
ncf_scores = ncf_metric_joint(res.numpy(), np.zeros(res.shape), modulation)

#Emprical Coverage for all values of alpha to see if pred_residual lies between +- qhat. 
alpha_levels = np.arange(0.05, 0.95, 0.1)
emp_cov_res = []
for alpha in tqdm(alpha_levels):
    qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
    prediction_sets = [- qhat*modulation, + qhat*modulation]
    emp_cov_res.append(emp_cov_joint(prediction_sets, pred_residual.numpy()))

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()

plot_name = plot_loc + '/coverage' + '_' + run.name + '.png'
plt.savefig(plot_name)
run.save_file(plot_name, 'output')
# %% 
###################################################################
#Filtering Sims using CP. 
def filter_sims_joint(prediction_sets, y_response):
    #Identifies the simulations that falls within the bounds.
    axes = tuple(np.arange(1,len(y_response.shape)))
    return ((y_response >= prediction_sets[0]).all(axis = axes) & (y_response <= prediction_sets[1]).all(axis = axes))

# %%
#Active Learning Pipeline - 3 kinds of acquisition functions.
#  
#CP - Filtering using Joint CP over the Residual Space. 
#PRE - Filtering using the predicitons with the largest residual errors. 
#RAND - Randomly sampling from the parameter space. 

funcs = ['CP', 'PRE', 'RAND']
n_iterations = 5
epochs = 100
acq_func = 'RAND' #CP, PRE, RAND
alpha = 0.5 #CP-alpha 
run.update_metadata({'threshold_alpha': alpha})


for acq_func in funcs:
    model = FNO_multi2d(configuration['T_in'], configuration['Step'], configuration['Modes'], configuration['Modes'], configuration['Variables'], configuration['Width_time'])
    model.load_state_dict(torch.load(trained_model, map_location=device))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
    
    test_mse = []
    sims_sampled = []

        
    for ii in range(n_iterations):

        #Prediction Residuals 
        params = lb + (ub - lb) * lhs(2, configuration['n_pred'])
        u_in_pred = gen_ic(params)
        pred_pred, mse, mae = validation_AR(model, u_in_pred, torch.zeros((u_in_pred.shape[0], u_in_pred.shape[1], u_in_pred.shape[2], T_out)), configuration['Step'], configuration['T_out'])
        pred_pred = pred_pred.permute(0,1,3,2)[:,0]
        uu_pred = pred_pred
        pred_residual = D(uu_pred)[...,1:-1, 1:-1]

        if acq_func == 'CP':
        #Selection/Rejection using Joint CP
            qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
            prediction_sets =  [- qhat*modulation, + qhat*modulation]
            filtered_sims = np.invert(filter_sims_joint(prediction_sets, pred_residual.numpy()))
            params_filtered = params[filtered_sims]
            print(f'{len(params_filtered)} predictions rejected')

        if acq_func == 'PRE':
        #Selection/Rejection using Descending order of PRE
            pred_residual_mean = torch.mean(torch.abs(pred_residual), axis=tuple(range(1, pred_residual.ndim)))
            pred_residual_mean_sorted, sort_index = torch.sort(pred_residual_mean)
            num_sims = int((1-alpha)*configuration['n_pred'])
            params_filtered = params[sort_index][:num_sims]
            print(f'{len(params_filtered)} predictions rejected')

        if acq_func == 'RAND':
        #Random Selection
            num_sims = int((1-alpha)*configuration['n_pred'])
            random_index = np.random.randint(0, configuration['n_pred'], num_sims)
            params_filtered = params[random_index]
            print(f'{len(params_filtered)} predictions selected')


        #Numerical Sim over the Rejected Predictions and adding to training data.
        x, t, u_sol = gen_data(params_filtered)
        u_sol_train = torch.vstack((u_sol_train, u_sol))
        train_loader = data_loader(u_sol_train)

        #Fine-tuning with the Sampled Sims. 
        start_time = default_timer()
        model = train(epochs, model, train_loader, test_loader, loss_func, optimizer, scheduler)
        train_time = default_timer() - start_time

        #Evaluation
        pred_test, mse, mae = validation_AR(model, test_a, test_u, step, T_out)
        test_mse.append(mse)
        print('Testing Error (MSE) : %.3e' % (mse))
        print('Testing Error (MAE) : %.3e' % (mae))
        print()

        test_mse.append(mse)
        sims_sampled.append(len(params_filtered))

    run.save_object(np.asarray(test_mse), 'output', name=acq_func + '_mse')
    run.save_object(np.asarray(sims_sampled), 'output', name=acq_func + '_sims_sampled')

run.close()
# %% 
plt.plot(test_mse)
plt.ylabel('MSE')
plt.xlabel('AL - Iterations')
plt.title(acq_func)
# %%
# #mse includes the first evaluation as well. 
# mse_cp = [0.0318748 , 0.00642189, 0.00386953, 0.00276095, 0.00239962, 0.0022232]
# sims_sampled_cp = [50, 77, 42, 20, 20]
# sims_sampled_pre = [50, 50, 50, 50, 50]
# mse_pre = [0.0318748 , 0.00637728, 0.00460812, 0.00302673, 0.00250596, 0.0022536]
# mse_rand = [0.0318748 , 0.00642819, 0.00465056, 0.00317558, 0.00264536,0.00237693]


# import matplotlib.pyplot as plt
# import numpy as np

# # Generate some sample data
# x = np.arange(1, 6)
# y1 = np.random.randint(1, 10, 5)
# y2 = np.random.randint(1, 10, 5)
# y3 = np.random.randint(1, 10, 5)

# # Define a neutral color scheme
# colors = ['#C44E52', '#55A868', '#4C72B0', '#CCB974', '#8172B3']

# # Create the plot
# plt.figure(figsize=(12, 8))

# # Plot three lines
# plt.plot(x, mse_cp[1:], marker='o', color=colors[0], label='CP')
# plt.plot(x, mse_pre[1:], marker='s', color=colors[2], label='PRE')
# plt.plot(x, mse_rand[1:], marker='^', color=colors[1], label='RAND')

# # Customize the plot
# plt.title('Active Learning Across 5 Training iterations.', fontsize=16)
# plt.xlabel('Iterations', fontsize=12)
# plt.xticks(x)
# plt.ylabel('MSE', fontsize=12)
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)

# # Set the background color to a white
# plt.gca().set_facecolor('white')

# # Show the plot
# plt.tight_layout()
# plt.show()
# # %%

# import matplotlib.pyplot as plt
# import numpy as np

# # Given data
# mse_cp = [0.00642189, 0.00386953, 0.00276095, 0.00239962, 0.0022232]
# sims_sampled_cp = [50, 77, 42, 20, 20]
# sims_sampled_pre = [50, 50, 50, 50, 50]
# mse_pre = [0.00637728, 0.00460812, 0.00302673, 0.00250596, 0.0022536]
# mse_rand = [0.00642819, 0.00465056, 0.00317558, 0.00264536, 0.00237693]

# # Calculate cumulative sum for x-axis
# x_cp = np.cumsum(sims_sampled_cp)
# x_pre = np.cumsum(sims_sampled_pre)

# # Create x-axis for RAND (assuming it's the same as PRE)
# x_rand = x_pre

# # Define a neutral color scheme
# colors = ['#C44E52', '#55A868', '#4C72B0']

# # Create the plot
# plt.figure(figsize=(12, 8))

# # Plot three lines
# plt.plot(x_cp, mse_cp, marker='o', color=colors[0], label='CP')
# plt.plot(x_pre, mse_pre, marker='s', color=colors[1], label='PRE')
# plt.plot(x_rand, mse_rand, marker='^', color=colors[2], label='RAND')

# # Customize the plot
# plt.title('Active Learning - Advection Equation', fontsize=16)
# plt.xlabel('Simulation Samples', fontsize=12)
# plt.ylabel('MSE', fontsize=12)
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)

# # Set the background color to white
# plt.gca().set_facecolor('white')


# combined_xticks = sorted(set(x_cp) | set(x_pre))
# # Set x-ticks to show the cumulative samples at each iteration
# plt.xticks(combined_xticks, [f'{x}' for x in combined_xticks])

# # Show the plot
# plt.tight_layout()
# plt.show()
# # %%
