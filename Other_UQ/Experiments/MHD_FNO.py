#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FNO modelled over the 2D Wave Equation auto-regressively 

Equation: u_tt = D*(u_xx + u_yy), D=1.0

"""

# %%
configuration = {"Case": 'MHD',
                 "Field": 'rho, u, v, p, Bx, By',
                 "Model": 'FNO',
                 "Epochs": 500,
                 "Batch Size": 5,
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
                 "Variables":6, 
                 "Loss Function": 'LP',
                 "Seed": 0, 
                 "UQ": 'Bayesian', #Deterministic, Dropout, Bayesian, Ensemble, SWAG
                 }

# %%
import os
from simvue import Run
with Run(mode='online') as run:
    run.init(folder="/Neural_PDE/Rebuttal", tags=['NPDE', 'FNO', 'PIUQ', 'AR', 'MHD', configuration['UQ']], metadata=configuration)

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
    sys.path.append(os.path.dirname(os.getcwd()))
    sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
    # %%
    #Importing the models and utilities. 
    from Neural_PDE.Utils.processing_utils import * 
    from Utils.training_utils import * 
    from Utils.loss_utils import * 
    # %% 
    #Setting up locations. 
    file_loc = os.getcwd()
    data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/Neural_PDE/Data'
    model_loc = file_loc + '/Weights'
    plot_loc = file_loc + '/Plots'
    #Setting up the seeds and devices
    seed = configuration['Seed']
    if configuration['UQ'] == 'Ensemble':
       seed = np.random.randint(1, 10000)
       run.update_metadata({'Seed': seed})

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # %%
    ################################################################
    # Loading Data 
    ################################################################

    # %%
    t1 = default_timer()
    data =  np.load(data_loc + '/Constrained_MHD_combined.npz')

    rho = data['rho'].astype(np.float32)[:, ::2]
    u = data['u'].astype(np.float32)[:, ::2]
    v = data['v'].astype(np.float32)[:, ::2]
    p = data['p'].astype(np.float32)[:, ::2]
    Bx = data['Bx'].astype(np.float32)[:, ::2]
    By  = data['By'].astype(np.float32)[:, ::2]

    x = data['x'].astype(np.float32)
    y = data['x'].astype(np.float32)
    dt = data['dt'].astype(np.float32)*2

    def stacked_fields(variables):
        stack = []
        for var in variables:
            var = torch.from_numpy(var) #Converting to Torch
            var = var.permute(0, 2, 3, 1) #Permuting to be BS, Nx, Ny, Nt
            stack.append(var)
        stack = torch.stack(stack, dim=1)
        return stack

    x_slice = 1 
    vars = stacked_fields([rho, u, v, p, Bx, By])[:, :, ::x_slice, ::x_slice, :]
    field = ['rho', 'u', 'v', 'p', 'Bx', 'By']

    ntrain = 200
    ntest = 200

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
    train_a = vars[:ntrain,...,:T_in]
    train_u = vars[:ntrain,...,T_in:T_out+T_in]

    test_a = vars[-ntest:,...,:T_in]
    test_u = vars[-ntest:,...,T_in:T_out+T_in]

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

    run.save_file(saved_normalisations, 'output')
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
    import Bayesian_Models as Models
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
    elif configuration['UQ']=='Ensemble':
        from Models.Base_FNO import *
        model = FNO_multi2d(T_in, step, modes, modes, num_vars, width_time)
    elif configuration['UQ']=='SWAG':
        from Models.Base_FNO import *
        from Utils.SWAG import *
        model = FNO_multi2d(T_in, step, modes, modes, num_vars, width_time)
        swag_model = SWAG(model)

    model.to(device)
    run.update_metadata({'Number of Params': int(model.count_params())})
    print("Number of model params : " + str(model.count_params()))

    #Setting up the optimizer and scheduler, loss and epochs 
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
    if configuration['UQ'] == 'SWAG':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=1.0)

    epochs = configuration['Epochs']

    loss_func = LpLoss(size_average=False)

    if configuration['UQ'] == 'Bayesian':
        loss_func = ELBO

    if configuration['UQ'] == 'MLE':
        loss_func = NLL

    # %%
    ####################################
    #Training Loop 
    ####################################

    if configuration['UQ']=='Deterministic':
        train_epoch = train_one_epoch
    elif configuration['UQ']=='Dropout':
        train_epoch = train_one_epoch
    elif configuration['UQ']=='Bayesian':
        train_epoch = train_one_epoch_bayesian
    elif configuration['UQ']=='MLE':
        train_epoch = train_one_epoch_MLE
    elif configuration['UQ']=='Ensemble':
        train_epoch = train_one_epoch
    elif configuration['UQ'] == 'SWAG':
        train_epoch = train_one_epoch
        swag_start = int(0.8*epochs)
        swag_freq = 10

    start_time = default_timer()
    for ep in range(epochs): #Training Loop - Epochwise

        model.train()
        t1 = default_timer()
        train_loss, test_loss = train_epoch(model, train_loader, test_loader, loss_func, optimizer, step, T_out)
        t2 = default_timer()

        train_loss = train_loss / ntrain / num_vars
        test_loss = test_loss / ntest / num_vars

        print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 3)}, Test Loss: {round(test_loss,3)}")
        run.log_metrics({'Train Loss': train_loss, 'Test Loss': test_loss})

        # Collect models for SWAG
        if configuration['UQ'] == 'SWAG':
            if ep >= swag_start and (ep - swag_start) % swag_freq == 0:
                swag_model.collect_model(model)
            
        scheduler.step()

    train_time = default_timer() - start_time

    # %%
    #Saving the Model
    saved_model = model_loc + '/' + configuration['Model'] + '_' + configuration['Case'] + '_' +run.name + '.pth'

    torch.save( model.state_dict(), saved_model)
    run.save_file(saved_model, 'output')

    if configuration['UQ'] == 'SWAG':
        saved_model = model_loc + '/' + configuration['Model'] + '_' + configuration['Case'] + '_' +run.name + '_swag.pt'
        swag_model.save(saved_model)
        run.save_file(saved_model, 'output')

    # %%
    #Validation
    if configuration['UQ']=='Deterministic':
        pred_set_encoded, mse, mae = validation(model, test_a, test_u_encoded, step, T_out)

    elif configuration['UQ']=='Dropout':
        pred_set_encoded, pred_set_encoded_var, mse, mae = validation_dropout(model, test_a, test_u_encoded, step, T_out, samples=20)

    elif configuration['UQ']=='Bayesian':
        pred_set_encoded, pred_set_encoded_var, mse, mae = validation_bayesian(model, test_a, test_u_encoded, step, T_out, samples=20)

    elif configuration['UQ']=='MLE':
        pred_set_encoded, pred_set_encoded_var, mse, mae = validation_MLE(model, test_a, test_u_encoded, step, T_out)

    if configuration['UQ']=='Ensemble':
        pred_set_encoded, mse, mae = validation(model, test_a, test_u_encoded, step, T_out)

    if configuration['UQ'] == 'SWAG':
        pred_set_encoded, pred_set_encoded_var, mse, mae = validation_SWAG(model, swag_model, test_a, test_u_encoded, step, T_out, samples=20)

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


        plot_name = plot_loc + '/' + field[var] + '_' + run.name + '.png'
        plt.savefig(plot_name)
        run.save_file(plot_name, 'output')

        try:

            pred_set_var = u_normalizer.decode(pred_set_encoded_var.to(device)).cpu()

            u_field = pred_set_var[idx][var]
                
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


            plot_name = plot_loc + '/' + field[var] + '_var_' + run.name + '.png'
            plt.savefig(plot_name)
            run.save_file(plot_name, 'output')
        except:
            pass

    run.close()
    # %%
