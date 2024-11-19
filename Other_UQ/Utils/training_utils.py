# %%
################################################################
# Training
################################################################

import numpy as np 
import torch 
from timeit import default_timer
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_grad_clip_norm = 10000.0   

def train_one_epoch(model, train_loader, test_loader, loss_func, optimizer, step, T_out):
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
            # y_old = y #Diff Loss

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

            
        #PI Loss
        # loss = loss_func(im)

        train_l2_step += loss.item()
        l2_full = loss_func(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_clip_norm, norm_type=2.0)
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


def validation(model, test_a, test_u, step, T_out):
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1,
                                            shuffle=False)
    pred_set = torch.zeros(test_u.shape)
    index = 0
    with torch.no_grad():
        for xx, yy in tqdm(test_loader, disable=False):
            xx, yy = xx.to(device), yy.to(device)
            t1 = default_timer()
            for t in range(0, T_out, step):
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


def validation_dropout(model, test_a, test_u, step, T_out, samples):
    model.enable_dropout()
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1,
                                            shuffle=False)
    pred_list = []
    with torch.no_grad():
        for ii in range(samples): 
            index = 0
            pred_set = torch.zeros(test_u.shape)
            for xx, yy in tqdm(test_loader, disable=True):
                xx, yy = xx.to(device), yy.to(device)
                t1 = default_timer()
                for t in range(0, T_out, step):
                    out = model(xx)

                    if t == 0:
                        pred = out
                    else:
                        pred = torch.cat((pred, out), -1)

                    xx = torch.cat((xx[..., step:], out), dim=-1)

                t2 = default_timer()
                pred_set[index] = pred
                index += 1
            pred_list.append(pred_set)
        pred_stack = torch.stack(pred_list)
        
        pred_mean = torch.mean(pred_stack, axis=0)
        pred_var = torch.std(pred_stack, axis=0)
        
        # print(t2 - t1)

        # Performance Metrics
        MSE_error = (pred_mean - test_u).pow(2).mean()
        MAE_error = torch.abs(pred_mean - test_u).mean()

    return pred_mean, pred_var, MSE_error, MAE_error


def validation_ensemble(models, test_a, test_u, step, T_out):
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1,
                                            shuffle=False)
    pred_list = []
    with torch.no_grad():
        for ii in range(len(models)): 
            model = models[ii]
            model = model.to(device)
            index = 0
            pred_set = torch.zeros(test_u.shape)
            for xx, yy in tqdm(test_loader, disable=True):
                xx, yy = xx.to(device), yy.to(device)
                t1 = default_timer()
                for t in range(0, T_out, step):
                    out = model(xx)

                    if t == 0:
                        pred = out
                    else:
                        pred = torch.cat((pred, out), -1)

                    xx = torch.cat((xx[..., step:], out), dim=-1)

                t2 = default_timer()
                pred_set[index] = pred
                index += 1
            pred_list.append(pred_set)
        pred_stack = torch.stack(pred_list)
        
        pred_mean = torch.mean(pred_stack, axis=0)
        pred_var = torch.std(pred_stack, axis=0)
        
        # print(t2 - t1)

        # Performance Metrics
        MSE_error = (pred_mean - test_u).pow(2).mean()
        MAE_error = torch.abs(pred_mean - test_u).mean()

    return pred_mean, pred_var, MSE_error, MAE_error

def train_one_epoch_MLE(model, train_loader, test_loader, loss_func, optimizer, step, T_out):
    model.enable_dropout()
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

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im[...,0:1]), dim=-1)


        train_l2_step += loss.item()
        l2_full = loss_func(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_clip_norm, norm_type=2.0)
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
 
                xx = torch.cat((xx[..., step:], out[...,0:1]), dim=-1)
            test_loss += loss_func(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    # t2 = default_timer()

    return train_loss, test_loss #remember to divide the ntrain/ntest and num_vars at the other end before logging.



def validation_MLE(model, test_a, test_u, step, T_out):
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1,
                                            shuffle=False)
    
    pred_set_mean, pred_set_log_var = torch.zeros(test_u.shape), torch.zeros(test_u.shape)

    index = 0
    with torch.no_grad():
        for xx, yy in tqdm(test_loader, disable=True):
            xx, yy = xx.to(device), yy.to(device)
            t1 = default_timer()
            for t in range(0, T_out, step):
                out = model(xx)

                if t == 0:
                    pred = out
                else:
                    pred = torch.cat((pred, out), -1)

                xx = torch.cat((xx[..., step:], out[...,0:1]), dim=-1)

            t2 = default_timer()
            pred_set_mean[index], pred_set_log_var[index] = pred[...,0:1], pred[...,1:2]
            pred_set_var = torch.mean(torch.exp(pred_set_log_var) + pred_set_mean**2, axis=0) - pred_set_mean**2

            index += 1
            # print(t2 - t1)

        # Performance Metrics
        MSE_error = (pred_set_mean - test_u).pow(2).mean()
        MAE_error = torch.abs(pred_set_mean - test_u).mean()

    return pred_set_mean, pred_set_var, MSE_error, MAE_error

# %% 
# Training loop

def train_one_epoch_bayesian(model, train_loader, test_loader, loss_func, optimizer, step, T_out):
    model.train()
    # t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        optimizer.zero_grad()
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        batch_size = xx.shape[0]

        for t in range(0, T_out, step):
            y = yy[..., t:t + step]
            im = model(xx)

            #ELBO
            loss += loss_func(model, im.reshape(batch_size, -1), 
                              y.reshape(batch_size, -1), 
                              batch_size)

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

        train_l2_step += loss.item()
        l2_full = torch.nn.MSELoss()(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=max_grad_clip_norm, norm_type=2.0)
        optimizer.step()

    train_loss = train_l2_full 

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
            test_loss += torch.nn.MSELoss()(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    # t2 = default_timer()

    return train_loss, test_loss #remember to divide the ntrain/ntest and num_vars at the other end before logging.


def validation_bayesian(model, test_a, test_u, step, T_out, samples):
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1,
                                            shuffle=False)
    pred_list = []
    with torch.no_grad():
        for ii in range(samples): 
            index = 0
            pred_set = torch.zeros(test_u.shape)
            for xx, yy in tqdm(test_loader, disable=True):
                xx, yy = xx.to(device), yy.to(device)
                t1 = default_timer()
                for t in range(0, T_out, step):
                    out = model(xx)

                    if t == 0:
                        pred = out
                    else:
                        pred = torch.cat((pred, out), -1)

                    xx = torch.cat((xx[..., step:], out), dim=-1)

                t2 = default_timer()
                pred_set[index] = pred
                index += 1
            pred_list.append(pred_set)
        pred_stack = torch.stack(pred_list)
        
        pred_mean = torch.mean(pred_stack, axis=0)
        pred_var = torch.std(pred_stack, axis=0)
        
        # print(t2 - t1)

        # Performance Metrics
        MSE_error = (pred_mean - test_u).pow(2).mean()
        MAE_error = torch.abs(pred_mean - test_u).mean()

    return pred_mean, pred_var, MSE_error, MAE_error


#SWAG

def validation_SWAG(model, swag_model, test_a, test_u, step, T_out, samples):
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1,
                                            shuffle=False)
    pred_list = []
    with torch.no_grad():
        for ii in range(samples): 
            swag_model.sample()
            index = 0
            pred_set = torch.zeros(test_u.shape)
            for xx, yy in tqdm(test_loader, disable=True):
                xx, yy = xx.to(device), yy.to(device)
                t1 = default_timer()
                for t in range(0, T_out, step):
                    out = model(xx)

                    if t == 0:
                        pred = out
                    else:
                        pred = torch.cat((pred, out), -1)

                    xx = torch.cat((xx[..., step:], out), dim=-1)

                t2 = default_timer()
                pred_set[index] = pred
                index += 1
            pred_list.append(pred_set)
        pred_stack = torch.stack(pred_list)
        
        pred_mean = torch.mean(pred_stack, axis=0)
        pred_var = torch.std(pred_stack, axis=0)
        
        # print(t2 - t1)

        # Performance Metrics
        MSE_error = (pred_mean - test_u).pow(2).mean()
        MAE_error = torch.abs(pred_mean - test_u).mean()

    return pred_mean, pred_var, MSE_error, MAE_error

# %% 
#Inductive CP using AER 
from Neural_PDE.UQ.inductive_cp import * 

def validation_AER(model, test_a, test_u, step, T_out, alpha):
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1,
                                            shuffle=False)
    pred_set = torch.zeros(test_u.shape)
    index = 0
    with torch.no_grad():
        for xx, yy in tqdm(test_loader, disable=True):
            xx, yy = xx.to(device), yy.to(device)
            t1 = default_timer()
            for t in range(0, T_out, step):
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

    ncf_scores = torch.abs(pred_set - test_u).numpy()
    qhat = calibrate(ncf_scores, len(pred_set), alpha)
    return pred_set, qhat, MSE_error, MAE_error


# %% 
#Inductive CP with PRE
from Utils.ConvOps_2d import * 
def validation_PRE(model, test_a, test_u, step, T_out, alpha, pre):
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1,
                                            shuffle=False)
    pred_set = torch.zeros(test_u.shape)
    index = 0
    with torch.no_grad():
        for xx, yy in tqdm(test_loader, disable=True):
            xx, yy = xx.to(device), yy.to(device)
            t1 = default_timer()
            for t in range(0, T_out, step):
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

    
    ncf_scores = torch.abs(pre.residual(pred_set.permute(0, 1, 4, 2, 3))).numpy()
    qhat = calibrate(ncf_scores, len(pred_set), alpha)
    return pred_set, qhat, MSE_error, MAE_error
