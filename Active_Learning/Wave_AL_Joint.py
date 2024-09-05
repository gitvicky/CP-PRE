#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uncertainty quantification of a Neural PDE Surrogate solving the Wave Equation using Physics Residuals and guaranteed using Conformal Prediction
Prediction Selection/Rejection based on CP bounds / PRE / Random Sampling 
Utilised for Active Learning

Eqn: u_tt = c**2 * (u_xx + u_yy)
Surrogate Model : FNO
Numerical Solver : https://github.com/gitvicky/Neural_PDE/blob/main/Numerical_Solvers/Wave/Wave_2D_Spectral.py
"""

# %% Training Configuration - used as the config file for simvue.
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
                 "T_in": 1,    
                 "T_out": 20,
                 "Step": 1,
                 "Width_time": 16, 
                 "Width_vars": 0,  
                 "Modes": 8,
                 "Variables":1, 
                 "Loss Function": 'LP',
                 "UQ": 'None', #None, Dropout
                 "n_train": 500,
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
import os
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
width_time = configuration['Width_time']
modes = configuration['Modes']
output_size = configuration['Step']
num_vars = configuration['Variables']
batch_size = configuration['Batch Size']

# %% 
#Simulation Setup
from pyDOE import lhs
from Neural_PDE.Numerical_Solvers.Wave.Wave_2D_Spectral import * 

Nx = 64 # Mesh Discretesiation 
x_min = -1.0 # Minimum value of x
x_max = 1.0 # maximum value of x
y_min = -1.0 # Minimum value of y 
y_max = 1.0 # Minimum value of y
tend = 1
c = 1.0 #Wave Speed <=1.0
t_slice = 5

sim = Wave_2D(Nx, x_min, x_max, tend, c)
dt, dx = sim.dt, sim.dx
dx , dt*t_slice
# %% 
#Utility Functions

# Utility Functions 
def gen_data(params):
    print("Generating Data via Numerical Sims.")
    u_sol = []
    for ii in tqdm(range(len(params))):
        x, y, t, u_soln = sim.solve(params[ii,0], params[ii,1], params[ii,2])
        u_sol.append(u_soln)

    #Extraction
    u_sol = np.asarray(u_sol)[:, ::t_slice]

    #Tensorize
    u = torch.tensor(u_sol, dtype=torch.float32)
    u = u.permute(0, 2, 3, 1)#BS, Nx, Ny, Ntx_min = -1.0 # Minimum value of x
    u = u.unsqueeze(1) #Adding the variable channel

    return x, dt, u

#Generate Initial Conditions
def gen_ic(params):
    u_ic = []
    for ii in tqdm(range(len(params))):
        sim.initialise(params[ii,0], params[ii,1], params[ii,2])
        u_ic.append(sim.vv)

    u_ic = np.asarray(u_ic)
    u_ic = torch.tensor(u_ic, dtype=torch.float32).unsqueeze(1).unsqueeze(-1)
    return u_ic

def normalisation(norm_strategy, data=None, norms=None):
    if norm_strategy == 'Min-Max':
        normalizer = MinMax_Normalizer
    elif norm_strategy == 'Range':
        normalizer = RangeNormalizer
    elif norm_strategy == 'Gaussian':
        normalizer = GaussianNormalizer
    elif norm_strategy == 'Identity':
        normalizer = Identity

    inputs = data[..., :T_in]
    outputs = data[..., T_in:T_out+T_in]
    in_normalizer = normalizer(inputs)
    out_normalizer = normalizer(outputs)

    if data==None:
        #Loading the Normaliation values
        in_normalizer = normalizer(torch.tensor(0))
        in_normalizer.a = torch.tensor(norms['in_a'])
        in_normalizer.b = torch.tensor(norms['in_b'])
        
        out_normalizer = normalizer(torch.tensor(0))
        out_normalizer.a = torch.tensor(norms['out_a'])
        out_normalizer.b = torch.tensor(norms['out_b'])
        
    return in_normalizer, out_normalizer


#Load Simulation data into Dataloader
def data_loader(uu, in_normalizer, out_normalizer, dataloader=True, shuffle=True):

    a = uu[..., :T_in]
    u = uu[..., T_in:T_out+T_in]

    # print("Input: " + str(a.shape))
    # print("Output: " + str(u.shape))

    #Performing the Normalisation and Setting up the DataLoaders
    a = in_normalizer.encode(a)
    u  = out_normalizer.encode(u)

    if dataloader:
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(a, u), batch_size=configuration['Batch Size'], shuffle=shuffle)
    #Performing the normalisation on the input alone. 
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
lb = np.asarray([10, 0.10, 0.10]) #Amplitude, x_pos, y_pos
ub = np.asarray([50, 0.50, 0.50]) #Amplitude, x_pos, y_pos

#Conv kernels --> Stencils 
from Utils.ConvOps_2d import ConvOperator
#Defining the required Convolutional Operations. 
D_tt = ConvOperator('t', 2)#, scale=alpha)
D_xx_yy = ConvOperator(('x','y'), 2)#, scale=beta)
#Additive Kernels 
D = ConvOperator()
c = torch.tensor(c, dtype=torch.float32)
D.kernel = D_tt.kernel - (c*dt/dx)**2 * D_xx_yy.kernel 
residual = D

# Residual
def residual(uu, boundary=False):
    res = D(uu)
    if boundary:
        return res
    else: 
        return res[...,1:-1,1:-1,1:-1]

################################################################
#Train Data
params = lb + (ub - lb) * lhs(3, configuration['n_train'])
x, t, u_sol_train = gen_data(params)
in_normalizer, out_normalizer = normalisation(configuration['Normalisation Strategy'], data=u_sol_train)

saved_normalisations = model_loc + '/' + configuration['Model'] + '_' + configuration['Case'] + '_' + run.name + '_' + 'norms.npz'

np.savez(saved_normalisations, 
        in_a=in_normalizer.a.numpy(), in_b=in_normalizer.b.numpy(), 
        out_a=out_normalizer.a.numpy(), out_b=out_normalizer.b.numpy()
        )

run.save_file(saved_normalisations, 'output')


train_loader = data_loader(u_sol_train, in_normalizer, out_normalizer)
test_loader = data_loader(u_sol_train[-10:], in_normalizer, out_normalizer) #Just kept to hefty evaluations each epoch. 

#Test Data
params = lb + (ub - lb) * lhs(3, configuration['n_test'])
x, t, u_sol = gen_data(params)
test_a, test_u = data_loader(u_sol, in_normalizer, out_normalizer, dataloader=False, shuffle=False)
test_mse = []

#Initialising the model
model = FNO_multi2d(configuration['T_in'], configuration['Step'], configuration['Modes'], configuration['Modes'], configuration['Variables'], configuration['Width_time'])
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

#Saving the Model
saved_model = model_loc + '/' + configuration['Model'] + '_' + configuration['Case'] + '_' +run.name + '_0' + '.pth'
torch.save( model.state_dict(), saved_model)
run.save_file(saved_model, 'output')

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
## Using Calibration Data from the Physics Residuals
params = lb + (ub - lb) * lhs(3, configuration['n_cal'])
u_in_cal = gen_ic(params)
u_in_cal = in_normalizer.encode(u_in_cal)
u_pred_cal, mse, mae = validation_AR(model, u_in_cal, torch.zeros((u_in_cal.shape[0], u_in_cal.shape[1], u_in_cal.shape[2], u_in_cal.shape[3], T_out)), step, T_out)

u_out_pred = out_normalizer.decode(u_pred_cal)
residual_pred_cal = residual(u_pred_cal.permute(0,1,4,2,3)[:,0])

# %%
#Prediction Residuals 
params = lb + (ub - lb) * lhs(3, configuration['n_pred'])
u_in_pred = gen_ic(params)
u_in_pred = in_normalizer.encode(u_in_pred)

pred_pred, mse, mae = validation_AR(model, u_in_pred, torch.zeros((u_in_pred.shape[0], u_in_pred.shape[1], u_in_pred.shape[2], u_in_pred.shape[3], T_out)), configuration['Step'], configuration['T_out'])
pred_pred =  out_normalizer.decode(pred_pred)

pred_pred = pred_pred.permute(0,1,4,2,3)[:,0]
uu_pred = pred_pred
pred_residual = residual(uu_pred)

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
n_iterations = 1
epochs = 10
acq_func = 'CP' #CP, PRE, RAND
alpha = 0.5 #CP-alpha 

for acq_func in funcs:
    model = FNO_multi2d(configuration['T_in'], configuration['Step'], configuration['Modes'], configuration['Modes'], configuration['Variables'], configuration['Width_time'])
    model.load_state_dict(torch.load(saved_model, map_location=device))
    test_mse = []
    for ii in range(n_iterations):

        #Prediction Residuals 
        params = lb + (ub - lb) * lhs(3, configuration['n_pred'])
        u_in_pred = gen_ic(params)
        u_in_pred = in_normalizer.encode(u_in_pred)
        pred_pred, mse, mae = validation_AR(model, u_in_pred, torch.zeros((u_in_pred.shape[0], u_in_pred.shape[1], u_in_pred.shape[2], u_in_pred.shape[3], T_out)), configuration['Step'], configuration['T_out'])
        pred_pred =  out_normalizer.decode(pred_pred)
        pred_pred = pred_pred.permute(0,1,4,2,3)[:,0]
        uu_pred = pred_pred
        pred_residual = residual(uu_pred)

        if acq_func == 'CP':
        #Selection/Rejection using Joint CP
            qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
            prediction_sets =  [- qhat*modulation, + qhat*modulation]
            filtered_sims = np.invert(filter_sims_joint(prediction_sets, pred_residual.numpy()))
            params_filtered = params[filtered_sims]
            print(f'{len(params_filtered)} predictions rejected')

        if acq_func == 'PRE':
        #Selection/Rejection using Descending order of PRE
            pred_residual_mean = torch.mean(pred_residual, axis=tuple(range(1, pred_residual.ndim)))
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
        train_loader = data_loader(u_sol_train, in_normalizer, out_normalizer)

        #Fine-tuning with the Sampled Sims. 
        start_time = default_timer()
        model = train(epochs, model, train_loader, test_loader, loss_func, optimizer, scheduler)
        train_time = default_timer() - start_time

        #Evaluation
        pred_test, mse, mae = validation_AR(model, test_a, test_u, step, T_out)
        test_mse.append(mse)
        print('Testing Error (MSE) : %.3e' % (mse))
        print('Testing Error (MAE) : %.3e' % (mae))

        #Saving the Model
        saved_model_AL = model_loc + '/' + configuration['Model'] + '_' + configuration['Case'] + '_' + run.name + '_' + acq_func + '_' + str(ii+1) + '.pth'
        torch.save( model.state_dict(), saved_model_AL)
        run.save_file(saved_model_AL, 'output')

    print(acq_func)
    print(np.asarray(test_mse))
    run.save_object(np.asarray(test_mse), 'output', name=acq_func)
