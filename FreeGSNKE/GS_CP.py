

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
"""
Performing PRE-CP over the GS equations with the data obtained from Tim's optimisations of PF coils
in FreeGSNKE 
"""

# %%
configuration = {"Case": 'Grad-Shafranov',
                 "Field": 'psi',
                 "Model": 'Conditional AE', #Siren or MFN
                 "Epochs": 500,
                 "Batch Size": 50, 
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GeLU',
                 "Physics Normalisation": 'No',
                 "Normalisation Strategy": 'Min-Max',
                 "Coords": 2, #Number of spatio-temporal coordinates - would form the number of inputs 
                 "Variables":1, #Number of variables being modelled - would form the number of outputs. 
                 "Context": 12,
                 "Loss Function": 'MSE',
                 "UQ": 'None', #None, Dropout
                 }

# %% 
#Importing the necessary packages
import sys
import numpy as np
from tqdm import tqdm 
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
import time 
from timeit import default_timer
from tqdm import tqdm 

#Adding the NPDE package to the system python path
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
# %%
#Importing the models and utilities. 
from Neural_PDE.Models.INR import *
from Neural_PDE.Utils.processing_utils import * 
from Neural_PDE.Utils.training_utils import * 

# %% 
#Settung up locations. 
file_loc = os.getcwd()
data_loc = os.getcwd()
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
import json
data_file = 'nunn_ixd_vignesh-2_8_0_sobol_512-icmldata.json'
# data_file = 'nunn_ixd_vignesh-2_8_0_sobol_1024-icmldata.json'
t1 = default_timer()
with open(data_file,  'r') as file:
    json_data = json.load(file)

pf_loc = []
psi = []
for ii in range(len(json_data)):
    pf_loc.append(json_data[ii]['x'])
    psi.append(np.asarray(json_data[ii]['psi']))

R = torch.tensor(np.asarray(json_data[ii]['R']), dtype=torch.float32)
Z = torch.tensor(np.asarray(json_data[ii]['Z']), dtype=torch.float32)
pf_loc = torch.tensor(np.asarray(pf_loc), dtype=torch.float32)
psi = torch.tensor(np.asarray(psi), dtype=torch.float32)

# %% 
#Setting up the Data for training 
RR, ZZ = torch.meshgrid(R, Z)
coords_input = torch.stack((RR, ZZ))
psi_output = psi.unsqueeze(1)

# %% 
from torch.utils.data import TensorDataset
#Normalisations. 
normalizer = MinMax_Normalizer
normalizer_RZ = normalizer(coords_input)
normalizer_psi = normalizer(psi_output)

coords_input = normalizer_RZ.encode(coords_input)
psi_output = normalizer_psi.encode(psi_output)

#Test-Train Split
pf_train = pf_loc[:300]
psi_train = psi_output[:300]
pf_test = pf_loc[300:]
psi_test = psi_output[300:]

train_loader = DataLoader(TensorDataset(pf_train, psi_train), batch_size=configuration['Batch Size'], shuffle=True) 
test_loader = DataLoader(TensorDataset(pf_test, psi_test), batch_size=configuration['Batch Size'], shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %%
################################################################
# Evaluation
###############################################################
def validation_INR(model, u0, uu):
    with torch.no_grad():
        inputs = u0.to(device)
        outputs = uu.to(device)
        coords = coords_input.repeat(inputs.shape[0], 1, 1, 1).to(device)
        print(inputs.shape, coords.shape)
        im = model(coords, inputs)
        pred = im
            
        # Performance Metrics
        MSE_error = (outputs - pred).pow(2).mean()
        MAE_error = torch.abs(outputs - pred).mean()

    return pred, MSE_error, MAE_error


# %% 

import torch.nn as nn
class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1)
        )
        
        # Feature integration
        fc_size = 64 * 33 * 33
        self.fc1 = nn.Linear(fc_size, 1024)
        self.fc2 = nn.Linear(1024 + 12, fc_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(8, out_channels, kernel_size=1)
        )
        
    def forward(self, x, additional_features):
        # Encode
        encoded = self.encoder(x)
        
        # Feature integration
        batch_size = x.size(0)
        flattened = encoded.view(batch_size, -1)
        fc1_out = torch.relu(self.fc1(flattened))
        combined = torch.cat([fc1_out, additional_features], dim=1)
        fc2_out = torch.relu(self.fc2(combined))
        
        # Reshape back to feature maps
        feature_maps = fc2_out.view(batch_size, 64, 33, 33)
        
        # Decode
        decoded = self.decoder(feature_maps)
        return decoded
    
    def count_params(self):
        nparams = 0

        for param in self.parameters():
            nparams += param.numel()
        return nparams

model = ConvAutoencoder(in_channels=2, out_channels=1).to(device)
model.load_state_dict(torch.load('Conditional AE_Grad-Shafranov_level-dingo.pth', map_location='cpu'))

print("Number of model params : " + str(model.count_params()))

#Setting up the optimizer and scheduler, loss and epochs 
optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
loss_func = torch.nn.MSELoss()
epochs = configuration['Epochs']

# %%
#Testing 
pred_set_encoded, mse, mae = validation_INR(model, pf_test, psi_test)

print('Testing Error (MSE) : %.3e' % (mse))
print('Testing Error (MAE) : %.3e' % (mae))

#Denormalising and reshaping the target and the predictions
pred_set = normalizer_psi.decode(pred_set_encoded.to(device)).cpu()
test_u = normalizer_psi.decode(psi_test.to(device)).cpu()

# %% 
#Plotting performance
import matplotlib as mpl

# Set matplotlib parameters
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['font.size'] = 24
mpl.rcParams['figure.figsize'] = (9,9)
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['axes.titlepad'] = 20
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['xtick.minor.size'] = 5.0
plt.rcParams['ytick.minor.size'] = 5.0
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.minor.width'] = 0.6
plt.rcParams['ytick.minor.width'] = 0.6
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '-'


idx = 0
u_field = test_u[idx][0].T
    
v_min = torch.min(u_field)
v_max = torch.max(u_field)


fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1)
pcm = ax.contourf(RR.T, ZZ.T, u_field, cmap=matplotlib.cm.coolwarm, vmin=v_min, vmax=v_max)
ax.title.set_text('Ground Truth')
fig.colorbar(pcm, pad=0.05)

u_field = pred_set[idx][0].T
ax = fig.add_subplot(1, 2, 2)
pcm = ax.contourf(RR.T, ZZ.T, u_field, cmap=matplotlib.cm.coolwarm,  vmin=v_min, vmax=v_max)
ax.title.set_text('Prediction')
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)

plt.savefig('GS_Pred.pdf',transparent=True, bbox_inches='tight')


# %%
#Setting up the PRE
from Utils.ConvOps_2d import ConvOperator
dz = Z[2] - Z[1]
dr = R[2] - R[1]

alpha, beta, gamma = 1,1,1
#Defining the required Convolutional Operations. 
D_R = ConvOperator(domain='x', order=1, scale=beta) 
D_RR = ConvOperator(domain='x', order=2, scale=gamma)
D_ZZ = ConvOperator(domain='y', order=2, scale=gamma)

def residual(psi, boundary=False, norms=True):

    if norms:
        res = D_RR(psi)*dz**2 - (1/R)*D_R(psi)*dz**2*dr + D_ZZ(psi)*dr**2
    else:
        res = D_RR(psi) - (1/R)*D_R(psi) + D_ZZ(psi)
    
    if boundary:
        return res
    else: 
        return res[...,1:-1,1:-1,1:-1]
    
# %%
#Residuals 
psi_pred = normalizer_psi.decode(psi_output)
ncal = int(0.5*len(psi_pred))
npred = int(0.5*len(psi_pred))
cal_psi = psi_pred[:ncal]
pred_psi = psi_pred[-npred:]

cal_residual = residual(cal_psi)
pred_residual = residual(pred_psi)
# %%
from Neural_PDE.UQ.inductive_cp import * 

#Marginal CP 
ncf_scores = np.abs(cal_residual.numpy())

#Emprical Coverage for all values of alpha to see if pred_residual lies between +- qhat. 
alpha_levels = np.arange(0.05, 0.95+0.1, 0.1)

emp_cov_res = []
for alpha in tqdm(alpha_levels):
    qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
    prediction_sets = [- qhat, + qhat]
    emp_cov_res.append(emp_cov(prediction_sets, pred_residual.numpy()))

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.legend()
# %%
import matplotlib as mpl 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import matplotlib.ticker as ticker

# Set matplotlib parameters
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['font.size'] = 24
mpl.rcParams['figure.figsize'] = (15,20)
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['axes.titlepad'] = 20
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['xtick.minor.size'] = 5.0
plt.rcParams['ytick.minor.size'] = 5.0
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.minor.width'] = 0.6
plt.rcParams['ytick.minor.width'] = 0.6
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '-'

qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=0.5)
idx = 50
u_field = pred_residual[idx].T
    
v_min = np.min(-qhat)
v_max = np.max(qhat)


fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 3, 1)
pcm = ax.contourf(RR[1:-1, 1:-1].T, ZZ[1:-1, 1:-1].T, u_field, cmap=matplotlib.cm.coolwarm, v_min=v_min, v_max=v_max)
plt.xlabel(r'$R$')
plt.ylabel(r'$Z$')
ax.title.set_text(r'PRE: $D_{GS}(\Psi)$')
fig.colorbar(pcm, pad=0.05)

u_field = -qhat.T
ax = fig.add_subplot(1, 3, 2)
pcm = ax.contourf(RR[1:-1, 1:-1].T, ZZ[1:-1, 1:-1].T, u_field, cmap=matplotlib.cm.coolwarm, v_min=v_min, v_max=v_max)
# plt.xlabel(r'$R$')
ax.title.set_text('Lower')
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)

u_field = qhat.T
ax = fig.add_subplot(1, 3, 3)
pcm = ax.contourf(RR[1:-1, 1:-1].T, ZZ[1:-1, 1:-1].T, u_field, cmap=matplotlib.cm.coolwarm, v_min=v_min, v_max=v_max)
# plt.xlabel(r'$R$')
ax.title.set_text('Upper')
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)

plt.savefig("Grad_Shaf_CP.svg", format="svg",transparent=True, bbox_inches='tight')
plt.savefig("Grad_Shaf_CP.pdf", format="pdf",transparent=True, bbox_inches='tight')
# %%
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import matplotlib.ticker as ticker

# Calculate v_min and v_max symmetrically from qhat
v_max = np.max(np.abs(qhat))  # Get the maximum absolute value
v_min = -v_max                # Set minimum to negative of maximum 

# Create figure and axes using subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 10))

# First subplot
pcm1 = ax1.contourf(RR[1:-1, 1:-1].T, ZZ[1:-1, 1:-1].T, pred_residual[idx].T,
                    cmap=matplotlib.cm.coolwarm, vmin=v_min, vmax=v_max)
ax1.set_xlabel(r'$R$')
ax1.set_ylabel(r'$Z$')
ax1.set_title(r'PRE: $D_{GS}(\Psi)$')
ax1.set_aspect('equal')

# Second subplot
pcm2 = ax2.contourf(RR[1:-1, 1:-1].T, ZZ[1:-1, 1:-1].T, -qhat.T,
                    cmap=matplotlib.cm.coolwarm, vmin=v_min, vmax=v_max)
ax2.set_title('Lower')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_aspect('equal')

# Third subplot
pcm3 = ax3.contourf(RR[1:-1, 1:-1].T, ZZ[1:-1, 1:-1].T, qhat.T,
                    cmap=matplotlib.cm.coolwarm, vmin=v_min, vmax=v_max)
ax3.set_title('Upper')
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_aspect('equal')

# Make space for colorbar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]

# Create colorbar that's linked to all three plots
norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.coolwarm), 
                    cax=cbar_ax)

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.3)
# # Save figures
# plt.savefig("Grad_Shaf_CP.svg", format="svg", transparent=True, bbox_inches='tight')
# plt.savefig("Grad_Shaf_CP.pdf", format="pdf", transparent=True, bbox_inches='tight')
# %%
#Filtering Sims -- using PRE only 
# res = cal_out_residual #Data-Driven
res = cal_residual #Physics-Driven

modulation = modulation_func(res.numpy(), np.zeros(res.shape))
ncf_scores = ncf_metric_joint(res.numpy(), np.zeros(res.shape), modulation)

#Emprical Coverage for all values of alpha to see if pred_residual lies between +- qhat. 
alpha_levels = np.arange(0.05, 0.95+0.1, 0.1)
emp_cov_res_joint = []
for alpha in tqdm(alpha_levels):
    qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
    prediction_sets = [- qhat*modulation, + qhat*modulation]
    emp_cov_res_joint.append(emp_cov_joint(prediction_sets, pred_residual.numpy()))

plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
plt.plot(1-alpha_levels, emp_cov_res_joint, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
plt.xlabel('1-alpha')
plt.ylabel('Empirical Coverage')
plt.title("Joint")
plt.legend()
# %%
import matplotlib as mpl
# Set matplotlib parameters
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['font.size'] = 24
mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['axes.titlepad'] = 20
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['xtick.minor.size'] = 5.0
plt.rcParams['ytick.minor.size'] = 5.0
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.minor.width'] = 0.6
plt.rcParams['ytick.minor.width'] = 0.6
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '-'
plt.figure()
plt.plot(1-alpha_levels, 1-alpha_levels, color='black', alpha=0.6, lw=5.0, label='Ideal')
plt.plot(1-alpha_levels, emp_cov_res, color='maroon', ls='--', alpha=0.6, lw=4.0, label='Marginal')
plt.plot(1-alpha_levels, emp_cov_res_joint, ls='-.', color='navy', alpha=0.8, lw=3.0, label='Joint')
plt.xlabel(r'1-$\alpha$', fontsize=36)
plt.ylabel(r'Empirical Coverage', fontsize=36)
plt.title('Coverage', fontsize=36)
plt.legend(fontsize=36)
plt.savefig('GS_coverage.svg', format="svg", bbox_inches='tight')
plt.savefig('GS_coverage.pdf', format="pdf", bbox_inches='tight')

# %%
