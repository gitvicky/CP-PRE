import numpy as np 
import torch 

data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/Neural_PDE/Data'

def stacked_fields(variables):
    stack = []
    for var in variables:
        var = torch.from_numpy(var) #Converting to Torch
        var = var.permute(0, 2, 3, 1) #Permuting to be BS, Nx, Ny, Nt
        stack.append(var)
    stack = torch.stack(stack, dim=1)
    return stack

def Wave(dist):
    if dist == 'in':
        data =  np.load(data_loc + '/Spectral_Wave_data_LHS.npz')
    else:
        data = np.load(data_loc + '/Spectral_Wave_data_LHS_OOD_halfspeed.npz')

    u_sol = data['u'].astype(np.float32)
    x = data['x'].astype(np.float32)
    y = data['y'].astype(np.float32)
    t = data['t'].astype(np.float32)
    u = torch.from_numpy(u_sol)
    u = u.permute(0, 2, 3, 1)
    u = torch.unsqueeze(u, 1)

    if dist=='in':
        ntrain = 500
        u = u[ntrain:]

    return x, y, t, u

def Navier_Stokes(dist):
    if dist == 'in':
        data =  np.load(data_loc + '/NS_Spectral_combined.npz')
    else: 
        data =  np.load(data_loc + '/NS_Spectral_combined_nu_1e-2_OOD.npz')
        # data = np.load(data_loc + '/NS_Spectral_IC_OOD.npz')

    u = data['u'].astype(np.float32)[:, ::2]
    v = data['v'].astype(np.float32)[:, ::2]
    p = data['p'].astype(np.float32)[:, ::2]
    # w = data['w'].astype(np.float32)[:, ::2]
    x = data['x'].astype(np.float32)
    y = data['x'].astype(np.float32)
    dt = data['dt'].astype(np.float32) * 2 
    
    vars = stacked_fields([u,v,p])

    if dist=='in':
        ntrain = 200
        u = u[ntrain:]

    return x, y, dt, vars

def MHD(dist):
    if dist == 'in':
        data =  np.load(data_loc + '/Constrained_MHD_combined.npz')
    else: 
        data =  np.load(data_loc + '/Constrained_MHD_combined_OOD.npz')

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


    if dist=='in':
        ntrain = 200
        u = u[ntrain:]

    return x, y, dt, vars


def Navier_Stokes_Incomp(n_sims):
    #PDEBench data
    data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/NOs_for_POs/Data'
    uv = np.load(data_loc + '/NS_incomp_velocity_100_128_128.npy') #uv being the two compnoents of velocity
    uv = torch.tensor(uv, dtype=torch.float) #Converting to tensor
    uv = uv.permute(0, 4, 2, 3, 1)[:n_sims]
    x, y = np.arange(0, 1, 128), np.arange(0, 1, 128) 
    t = np.arange(0, 5.0, 0.005)
    dt = 0.005
    dt = torch.tensor(dt, dtype=torch.float)

    return x,  y, dt, uv


def JOREK(n_sims):
    #JOREK MultiBlob Data 
    #https://iopscience.iop.org/article/10.1088/1741-4326/ad313a/meta
    data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/NOs_for_POs/Data'
    data = data_loc + '/JOREK_filtered.npz' 

    rho = np.load(data)['rho'].astype(np.float32)[:n_sims] / 1e20
    phi = np.load(data)['Phi'].astype(np.float32)[:n_sims] / 1e5
    T = np.load(data)['T'].astype(np.float32)[:200][:n_sims] / 1e6

    rho = np.nan_to_num(rho)
    phi = np.nan_to_num(phi)
    T = np.nan_to_num(T)

    fields = stacked_fields([rho,phi,T])

    x_grid =  np.load(data)['Rgrid'].astype(np.float32)
    y_grid =  np.load(data)['Zgrid'].astype(np.float32)
    t_grid =  np.load(data)['time'].astype(np.float32)

    t_norm = t_grid / t_grid[-1]
    dt = t_norm[1] - t_norm[0]
    dt = torch.tensor(dt, dtype=torch.float)

    return x_grid, y_grid, dt, fields
