#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluating other UQ methods 

1. MC Dropout
2. Deep Ensembles 
3. Last Layer Bayesian

"""

# %%
import yaml
config_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/Residuals_UQ/Other_UQ/Evaluation/Configs/'
config = config_loc + 'NS_FNO.yaml'
configuration = yaml.safe_load(open(config))

# print('UQ Method: ' +  configuration['UQ'])
print(configuration)
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

#Adding the NPDE package to the system python path
import sys
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
# %%
#Importing the models and utilities. 
from Neural_PDE.Utils.processing_utils import * 
from Utils.training_utils import * 
from Utils.loss_utils import * 
from Neural_PDE.UQ.inductive_cp import * 
from Utils.ConvOps_2d import *
from PRE_estimations import * 

# %% 
#Setting up locations. 
file_loc = os.getcwd()
data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/Neural_PDE/Data'
model_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/Residuals_UQ/Other_UQ/Experiments/Weights'
plot_loc = file_loc + '/Plots'
#Setting up the seeds and devices
seed = configuration['Seed']
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
################################################################
# Loading Data 
################################################################
from data_loaders import *

pde = configuration['Case']
if pde == 'Wave':
    x, y, t, vars = Wave(configuration['Dist'])
    dx, dt = x[1]-x[0], t[1]-t[0]
    # ntrain = 500
    ntest = 300
    pre = PRE_Wave(dt, dx)
    
if pde == 'Navier-Stokes':
    x, y, dt, vars = Navier_Stokes(configuration['Dist'])
    dx, dy = x[1]-x[0], y[1]-y[0]
    # ntrain = 200
    ntest = 300
    pre = PRE_NS(dt, dx, dy)

if pde == 'MHD':
    x, y, dt, vars = MHD(configuration['Dist'])
    dx, dy = x[1]-x[0], y[1]-y[0]
    # ntrain = 200
    ntest = 300
    pre = PRE_MHD(dt, dx, dy)

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

# #Setting up train and test
# train_a = vars[:ntrain,...,:T_in]
# train_u = vars[:ntrain,...,T_in:T_out+T_in]


# %% 


uqs = configuration['UQ']

for jj in range(len(uqs)):

    configuration['UQ'] = uqs[jj]

    if configuration['UQ'] == 'Ensemble':
        run_name = configuration['Runs'][configuration['UQ']][0]
    else: 
        run_name = configuration['Runs'][configuration['UQ']]


    l2 = []
    coverage = []
    eval_time = []

    for ii in range(10):
        test =  vars[np.random.choice(len(vars), size=ntest, replace=False)]  

        test_a = test[...,:T_in]
        test_u = test[...,T_in:T_out+T_in]

        # print("Testing Input: " + str(test_a.shape))
        # print("Testing Output: " + str(test_u.shape))

        ################################################################
        # Normalisation
        ################################################################

        norm_strategy = configuration['Normalisation Strategy']

        def normalisation(norm_strategy, norms):
            if norm_strategy == 'Min-Max':
                normalizer = MinMax_Normalizer
            elif norm_strategy == 'Range':
                normalizer = RangeNormalizer
            elif norm_strategy == 'Gaussian':
                normalizer = GaussianNormalizer
            elif norm_strategy == 'Identity':
                normalizer = Identity

            #Loading the Normaliation values
            in_normalizer = MinMax_Normalizer(torch.tensor(0))
            in_normalizer.a = torch.tensor(norms['in_a'])
            in_normalizer.b = torch.tensor(norms['in_b'])

            out_normalizer = MinMax_Normalizer(torch.tensor(0))
            out_normalizer.a = torch.tensor(norms['out_a'])
            out_normalizer.b = torch.tensor(norms['out_b'])
            
            return in_normalizer, out_normalizer

        norms = np.load(model_loc + '/FNO_'+configuration['Case']+'_'+run_name+'_norms.npz')
        a_normalizer, u_normalizer = normalisation(configuration['Normalisation Strategy'], norms)

        # train_a = a_normalizer.encode(train_a)
        test_a = a_normalizer.encode(test_a)
        # train_u = u_normalizer.encode(train_u)
        test_u_encoded = u_normalizer.encode(test_u)

        
        ################################################################
        # Loading the Model
        ################################################################

        if configuration['UQ']=='Deterministic':
            from Models.Base_FNO import *
            model = FNO_multi2d(T_in, step, modes, modes, num_vars, width_time)
        elif configuration['UQ']=='Dropout':
            from Models.Dropout_FNO import *
            model = FNO_multi2d_Dropout(T_in, step, modes, modes, num_vars, width_time)
        elif configuration['UQ'] == 'Bayesian':
            from Models.Bayesian_FNO import *
            model = FNO_multi2d_Bayesian(T_in, step, modes, modes, num_vars, width_time)
        elif configuration['UQ']=='MLE':
            from Models.Base_FNO import *
            model = FNO_multi2d(T_in, step*2, modes, modes, num_vars, width_time)
        elif configuration['UQ']=='AER':
            from Models.Base_FNO import *
            model = FNO_multi2d(T_in, step, modes, modes, num_vars, width_time)
        elif configuration['UQ']=='PRE':
            from Models.Base_FNO import *
            model = FNO_multi2d(T_in, step, modes, modes, num_vars, width_time)

        try:
            model.load_state_dict(torch.load(model_loc + '/FNO_' + configuration['Case'] + '_' + run_name + '.pth', map_location='cpu'))
        except:
            pass

        if configuration['UQ']=='Ensemble':
            from Models.Base_FNO import *
            models = []
            for ii in range(len(configuration['Runs'][configuration['UQ']])):
                run_name = configuration['Runs'][configuration['UQ']][ii]
                model = FNO_multi2d(T_in, step, modes, modes, num_vars, width_time)
                model.load_state_dict(torch.load(model_loc + '/FNO_' + configuration['Case'] + '_' + run_name + '.pth', map_location='cpu'))
                models.append(model)


        if configuration['UQ'] == 'SWAG':
            from Models.Base_FNO import *
            from Utils.SWAG import *

            model = FNO_multi2d(T_in, step, modes, modes, num_vars, width_time)
            swag_model =  SWAG.load(model_loc + '/FNO_' + configuration['Case'] + '_' + run_name + '_swag.pt', model)

        model.to(device)

        # print("Number of model params : " + str(model.count_params()))

        
        ################################################################
        #Validation
        ################################################################
        t1 = default_timer()

        if configuration['UQ']=='Deterministic':
            pred_set_encoded, mse, mae = validation(model, test_a, test_u_encoded, step, T_out)

        elif configuration['UQ']=='Dropout':
            pred_set_encoded, pred_set_encoded_std, mse, mae = validation_dropout(model, test_a, test_u_encoded, step, T_out, samples=5)

        elif configuration['UQ']=='Bayesian':
            pred_set_encoded, pred_set_encoded_std, mse, mae = validation_bayesian(model, test_a, test_u_encoded, step, T_out, samples=5)

        elif configuration['UQ']=='MLE':
            pred_set_encoded, pred_set_encoded_std, mse, mae = validation_MLE(model, test_a, test_u_encoded, step, T_out)

        if configuration['UQ']=='Ensemble':
            pred_set_encoded, pred_set_encoded_std, mse, mae = validation_ensemble(models, test_a, test_u_encoded, step, T_out)

        if configuration['UQ'] == 'SWAG':
                pred_set_encoded, pred_set_encoded_std, mse, mae = validation_SWAG(model, swag_model, test_a, test_u_encoded, step, T_out, samples=5)

        if configuration['UQ']=='AER':
            alpha = 0.05
            pred_set_encoded, qhat,  mse, mae = validation_AER(model, test_a, test_u_encoded, step, T_out, alpha)

        if configuration['UQ']=='PRE':
            alpha=0.05
            pred_set_encoded, qhat, mse, mae = validation_PRE(model, test_a, test_u_encoded, step, T_out, alpha, pre)

        t2 = default_timer()

        # print('Testing Error (MSE) : %.3e' % (mse))
        # print('Testing Error (MAE) : %.3e' % (mae))
        # print(f'Evaluation Time (s) : {t2-t1}')

        try:
            pred_set_encoded_std
            pred_sets = [pred_set_encoded.numpy() - 2*pred_set_encoded_std.numpy(), pred_set_encoded.numpy() + 2*pred_set_encoded_std.numpy()]
            marginal_cov = emp_cov(pred_sets, test_u_encoded.numpy())
            joint_cov = emp_cov_joint(pred_sets, test_u_encoded.numpy())

            # print(f'Marginal Coverage (%) : {marginal_cov}')
            # print(f'Joint Coverage (%) : {joint_cov}')
        except: 
            pass

        if configuration['UQ'] == 'AER' :
            pred_sets = [pred_set_encoded.numpy() - qhat, pred_set_encoded.numpy() + qhat]
            marginal_cov = emp_cov(pred_sets, test_u_encoded.numpy())
            joint_cov = emp_cov_joint(pred_sets, test_u_encoded.numpy())

            # print(f'Marginal Coverage (%) : {marginal_cov}')
            # print(f'Joint Coverage (%) : {joint_cov}')

        if configuration['UQ'] == 'PRE':
            pred_sets = [-qhat, qhat]
            marginal_cov = emp_cov(pred_sets, pre.residual(pred_set_encoded.permute(0, 1, 4, 2, 3)).numpy())
            # joint_cov = emp_cov_joint(pred_sets, test_u_encoded.numpy())
            # print(f'Marginal Coverage (%) : {marginal_cov}')
            # print(f'Joint Coverage (%) : {joint_cov}')

        #Denormalising the predictions
        pred_set = u_normalizer.decode(pred_set_encoded.to(device)).cpu()

        l2.append(mse)
        coverage.append(marginal_cov)
        eval_time.append(t2-t1)
        
        ################################################################
        # Plotting
        ################################################################

        def plotting(idx=10):
            idx = np.random.randint(0,ntest) 


            for var in range(num_vars):

                u_field = test_u[idx][var]
                    
                v_min_1 = torch.min(u_field[...,0])
                v_max_1 = torch.max(u_field[...,0])

                v_min_2 = torch.min(u_field[..., int(T_out/ 2)])
                v_max_2 = torch.max(u_field[..., int(T_out/ 2)])

                v_min_3 = torch.min(u_field[..., -1])
                v_max_3 = torch.max(u_field[..., -1])

                fig = plt.figure(figsize=plt.figaspect(0.5))
                ax = fig.add_subplot(2, 3, 1)
                pcm = ax.imshow(u_field[..., 0], cmap=matplotlib.cm.coolwarm, extent=[0.0, 1.0, 0.0, 1.0], vmin=v_min_1, vmax=v_max_1)
                # ax.title.set_text('Initial')
                ax.title.set_text('t=' + str(T_in))
                ax.set_ylabel('Solution')
                fig.colorbar(pcm, pad=0.05)

                ax = fig.add_subplot(2, 3, 2)
                pcm = ax.imshow(u_field[..., int(T_out/ 2)], cmap=matplotlib.cm.coolwarm, extent=[0.0, 1.0, 0.0, 1.0], vmin=v_min_2,
                                vmax=v_max_2)
                # ax.title.set_text('Middle')
                ax.title.set_text('t=' + str(int((T_out+ T_in) / 2)))
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                fig.colorbar(pcm, pad=0.05)

                ax = fig.add_subplot(2, 3, 3)
                pcm = ax.imshow(u_field[..., -1], cmap=matplotlib.cm.coolwarm, extent=[0.0, 1.0, 0.0, 1.0], vmin=v_min_3, vmax=v_max_3)
                # ax.title.set_text('Final')
                ax.title.set_text('t=' + str(T_out+ T_in))
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                fig.colorbar(pcm, pad=0.05)

                u_field = pred_set[idx][var]

                ax = fig.add_subplot(2, 3, 4)
                pcm = ax.imshow(u_field[..., 0], cmap=matplotlib.cm.coolwarm, extent=[0.0, 1.0, 0.0, 1.0], vmin=v_min_1, vmax=v_max_1)
                ax.set_ylabel('FNO')

                fig.colorbar(pcm, pad=0.05)

                ax = fig.add_subplot(2, 3, 5)
                pcm = ax.imshow(u_field[..., int(T_out/ 2)], cmap=matplotlib.cm.coolwarm, extent=[0.0, 1.0, 0.0, 1.0], vmin=v_min_2,
                                vmax=v_max_2)
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                fig.colorbar(pcm, pad=0.05)

                ax = fig.add_subplot(2, 3, 6)
                pcm = ax.imshow(u_field[..., -1], cmap=matplotlib.cm.coolwarm, extent=[0.0, 1.0, 0.0, 1.0], vmin=v_min_3, vmax=v_max_3)
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                fig.colorbar(pcm, pad=0.05)


                try:

                    pred_set_std = u_normalizer.decode(pred_set_encoded_std.to(device)).cpu()

                    u_field = pred_set_std[idx][var]
                        
                    v_min_1 = torch.min(u_field[..., 0])
                    v_max_1 = torch.max(u_field[..., 0])

                    v_min_2 = torch.min(u_field[..., int(T_out/ 2)])
                    v_max_2 = torch.max(u_field[..., int(T_out/ 2)])

                    v_min_3 = torch.min(u_field[..., -1])
                    v_max_3 = torch.max(u_field[..., -1])

                    fig = plt.figure(figsize=plt.figaspect(0.5))
                    ax = fig.add_subplot(1, 3, 1)
                    pcm = ax.imshow(u_field[..., 0], cmap=matplotlib.cm.coolwarm, extent=[0.0, 1.0, 0.0, 1.0], vmin=v_min_1, vmax=v_max_1)
                    # ax.title.set_text('Initial')
                    ax.title.set_text('t=' + str(T_in))
                    ax.set_ylabel('Solution')
                    fig.colorbar(pcm, pad=0.05)

                    ax = fig.add_subplot(1, 3, 2)
                    pcm = ax.imshow(u_field[..., int(T_out/ 2)], cmap=matplotlib.cm.coolwarm, extent=[0.0, 1.0, 0.0, 1.0], vmin=v_min_2,
                                    vmax=v_max_2)
                    # ax.title.set_text('Middle')
                    ax.title.set_text('t=' + str(int((T_out+ T_in) / 2)))
                    ax.axes.xaxis.set_ticks([])
                    ax.axes.yaxis.set_ticks([])
                    fig.colorbar(pcm, pad=0.05)

                    ax = fig.add_subplot(1, 3, 3)
                    pcm = ax.imshow(u_field[..., -1], cmap=matplotlib.cm.coolwarm, extent=[0.0, 1.0, 0.0, 1.0], vmin=v_min_3, vmax=v_max_3)
                    # ax.title.set_text('Final')
                    ax.title.set_text('t=' + str(T_out+ T_in))
                    ax.axes.xaxis.set_ticks([])
                    ax.axes.yaxis.set_ticks([])
                    fig.colorbar(pcm, pad=0.05)

                except:
                    pass

    l2 = np.asarray(l2)
    coverage = np.asarray(coverage)
    eval_time = np.asarray(eval_time)

    l2_mean, l2_std = np.mean(l2), np.std(l2)
    cov_mean, cov_std = np.mean(coverage)*100, np.std(coverage)*100
    eval_mean, eval_std = np.mean(eval_time), np.std(eval_time)


    print('')
    print('UQ Method: ' +  configuration['UQ'])
    print(f"L2 Mean: {l2_mean:.2e}, L2 STD: {l2_std:.2e}")
    print(f"Cov. Mean: {cov_mean:.2f}, Cov. STD: {cov_std:.2f}")
    print(f'Eval. Time: {eval_mean:.1f}, Eval. STD: {eval_std:.1f}')
    print('')
    print('')

# %%