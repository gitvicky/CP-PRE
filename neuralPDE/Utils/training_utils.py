# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2 May 2024 
@author: @vgopakum

Training and Inference pipelines for Neural-PDE solvers with autoregresive temporal rollouts. Data shape - [Batch, variables, Nx, Ny, Nt]
!!!!!!! currently devised for the FNOs but should be suited to work for U-Nets as well - basically how the time is kept together. 
"""

# %%
################################################################
# Training - Autoregressive temporal rollouts. 
################################################################

import numpy as np 
import torch 
from timeit import default_timer
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_one_epoch_AR(model, train_loader, test_loader, loss_func, optimizer, step, T_out):
    model.train()
    # t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        optimizer.zero_grad()
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        y_old = xx[..., -step:]
        batch_size = xx.shape[0]

        for t in range(0, T_out, step):
            y = yy[..., t:t + step]
            im = model(xx)

            #Recon Loss
            loss += loss_func(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            #Diff Loss https://iopscience.iop.org/article/10.1088/1741-4326/ad313a/pdf
            # pred_diff = im - xx[..., -step:]
            # y_diff = y - y_old
            # loss += loss_func(pred_diff.reshape(batch_size, -1), y_diff.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)
            # y_old = y #Diff Loss

            
        #PI Loss
        # loss = loss_func(im)

        train_l2_step += loss.item()
        l2_full = loss_func(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        loss.backward()
        # torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=max_grad_clip_norm, norm_type=2.0)
        optimizer.step()

    train_loss = train_l2_full 
    # train_l2_step = train_l2_step / ntrain / (T_out / step) /num_vars

    # Validation Loop
    test_loss = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            xx, yy = xx.to(device), yy.to(device)
            batch_size = xx.shape[0]

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


################################################################
# Validation / Inference - Autoregressive Temporal rollouts. 
################################################################
def validation_AR(model, test_a, test_u, step, T_out):
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1,
                                            shuffle=False)
    pred_set = torch.zeros(test_u.shape)
    index = 0
    with torch.no_grad():
        for xx, yy in tqdm(test_loader):
            loss = 0
            xx, yy = xx.to(device), yy.to(device)
            t1 = default_timer()
            for t in range(0, T_out, step):
                y = yy[..., t:t + step]
                out = model(xx)

                if t == 0:
                    pred = out
                else:
                    pred = torch.cat((pred, out), -1)

                xx = torch.cat((xx[..., step:], out), dim=-1)

            t2 = default_timer()
            pred_set[index] = pred
            index += 1
            # print(t2 - t1)

        # Performance Metrics
        MSE_error = (pred_set - test_u).pow(2).mean()
        MAE_error = torch.abs(pred_set - test_u).mean()

    return pred_set, MSE_error, MAE_error

# %% 
################################################################
# Training - With PDE Losses
################################################################

def train_one_epoch_PINN(model, train_loader, test_loader, loss_func, optimizer):
    model.train()
    train_loss = 0
    for xx, yy in train_loader:
        optimizer.zero_grad()
        xx = xx.to(device)
        yy = yy.to(device)
        batch_size = xx.shape[0]

        loss = loss_func(model, xx)
 
        loss.backward()
        # torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=max_grad_clip_norm, norm_type=2.0)
        optimizer.step()

        train_loss += loss.item()
    
    # Validation Loop
    test_loss = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            xx, yy = xx.to(device), yy.to(device)
            batch_size = xx.shape[0]
            out, _ = model(xx)
            test_loss += (out.reshape(batch_size, -1) -  yy.reshape(batch_size, -1)).pow(2).mean().item()

    return train_loss, test_loss #remember to divide the ntrain/ntest and num_vars at the other end before logging.

# %% 
################################################################
# Training - Coordinate based MLPs (INRs) with context
################################################################
def train_one_epoch_INR(model, train_loader, test_loader, loss_func, optimizer):
    model.train()
    train_loss = 0
    for xx, yy in train_loader:
        optimizer.zero_grad()
        xx, yy = xx.to(device), yy.to(device)

        
        out, _ = model(xx)
        loss = loss_func(out, yy)

        loss.backward()
        # torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=max_grad_clip_norm, norm_type=2.0)
        optimizer.step()

        train_loss += loss.item()
    
    # Validation Loop
    test_loss = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            xx, yy = xx.to(device), yy.to(device)

            out, _ = model(xx)
            test_loss += (out.reshape(xx.shape[0], -1) -  yy.reshape(xx.shape[0], -1)).pow(2).mean().item()

    return train_loss, test_loss #remember to divide the ntrain/ntest and num_vars at the other end before logging.

# %%
################################################################
# Training - Supervised with no roll-outs. 
################################################################

def train_one_epoch(model, train_loader, test_loader, loss_func, optimizer):
    model.train()
    train_loss = 0
    for xx, yy in train_loader:
        optimizer.zero_grad()
        xx = xx.to(device)
        yy = yy.to(device)
        batch_size = xx.shape[0]

        im = model(xx)
        loss = loss_func(im.reshape(batch_size, -1), yy.reshape(batch_size, -1))
 
        loss.backward()
        # torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=max_grad_clip_norm, norm_type=2.0)
        optimizer.step()

        train_loss += loss.item()
    

    # Validation Loop
    test_loss = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            xx, yy = xx.to(device), yy.to(device)
            batch_size = xx.shape[0]
            out = model(xx)
            test_loss += loss_func(out.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    return train_loss, test_loss #remember to divide the ntrain/ntest and num_vars at the other end before logging.


# %%
################################################################
# Training - Supervised with no roll-outs. -- DeepOnets with a Trunk and Branch Input 
################################################################

def train_one_epoch_don(model, train_loader, test_loader, loss_func, optimizer):
    model.train()
    train_loss = 0
    for br, tr, yy in train_loader:
        optimizer.zero_grad()
        br = br.to(device)
        tr = tr.to(device)
        xx = [br, tr]
        yy = yy.to(device)
        batch_size = yy.shape[0]

        im = model(xx)
        loss = loss_func(im.reshape(batch_size, -1), yy.reshape(batch_size, -1))
 
        loss.backward()
        # torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=max_grad_clip_norm, norm_type=2.0)
        optimizer.step()

        train_loss += loss.item()
    
    # Validation Loop
    test_loss = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            br = br.to(device)
            tr = tr.to(device)
            xx = [br, tr]
            yy = yy.to(device)
            batch_size = xx.shape[0]
            out = model(xx)
            test_loss += loss_func(out.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    return train_loss, test_loss #remember to divide the ntrain/ntest and num_vars at the other end before logging.


# %% 
################################################################
# Validation / Inference for INR / PINNs 
################################################################
def validation_INR(model, test_a, test_u):
    with torch.no_grad():
        test_a, test_u = test_a.to(device), test_u.to(device)
        out, _ = model(test_a)

        # Performance Metrics
        MSE_error = (out - test_u).pow(2).mean()
        MAE_error = torch.abs(out - test_u).mean()

        return out, MSE_error, MAE_error


def validation(model, test_loader):
    mean_sq_err = 0
    mean_abs_err = 0
    outs = []
    with torch.no_grad():
        for xx, yy in test_loader:
            xx, yy = xx.to(device), yy.to(device)
            out = model(xx)
            outs.append(out)

            # Performance Metrics
            mean_sq_err += (out - yy).pow(2).mean()
            mean_abs_err = torch.abs(out - yy).mean()

        return torch.stack(outs), mean_sq_err, mean_abs_err