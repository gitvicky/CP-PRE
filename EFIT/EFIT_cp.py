# %% 
import os 
import numpy as np
import pandas as pd 
from tqdm import tqdm

import sys 
sys.path.append("..")
from Neural_PDE.UQ.inductive_cp import * 
from matplotlib import pyplot as plt 

# %% 
#Loading the wall and mesh
df_jet_wall = pd.read_csv(
    "jet_wall.csv",
    header=None,
    delim_whitespace=True,
)


df_psi_coords = pd.read_csv(
    "psi_coordinates.csv",
)

R = df_psi_coords.PSIR
Z = df_psi_coords.PSIZ

R_mesh, Z_mesh = np.meshgrid(R, Z)

dr = R[1] - R[0]
dz = Z[1] - Z[0]

R, Z = torch.tensor(R, dtype=torch.float32), torch.tensor(Z, dtype=torch.float32)
# %%
#Loading the data
df_equilibria = pd.read_csv("v_results.csv",header=None,index_col=0,nrows=10000)
df_equilibria.columns = (
    [f"psi_true_{i+1}" for i in range(1089)]
    + [f"psi_pred_{i+1}" for i in range(1089)]
    + [f"pff_true_{i+1}" for i in range(1089)]
    + [f"pff_pred_{i+1}" for i in range(1089)]
    + [f"gs_true_{i+1}" for i in range(1089)]
    + [f"gs_pred_{i+1}" for i in range(1089)]
)
# %%
psi_true = []
psi_pred = []
for ii in tqdm(range(len(df_equilibria))):
    psi_true.append(np.asarray(df_equilibria.iloc[ii][[f"psi_true_{i+1}" for i in range(1089)]]).reshape(33,33))
    psi_pred.append(np.asarray(df_equilibria.iloc[ii][[f"psi_pred_{i+1}" for i in range(1089)]]).reshape(33,33))

psi_true, psi_pred  = np.asarray(psi_true), np.asarray(psi_pred)
psi_true, psi_pred  = torch.tensor(psi_true, dtype=torch.float32).unsqueeze(1), torch.tensor(psi_pred, dtype=torch.float32).unsqueeze(1)
# %%
#Setting up the PRE
from Utils.ConvOps_2d import ConvOperator
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
ncal = int(0.6*len(psi_pred))
npred = int(0.3*len(psi_pred))
cal_psi = psi_pred[:ncal]
pred_psi = psi_pred[-npred:]

#Residuals 
cal_residual = residual(cal_psi)
pred_residual = residual(pred_psi)
# %%

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
# %% 
from Utils.plot_tools import subplots_2d
#Plotting the fields, prediction, abs error and the residual
idx = 250
values = [pred_psi[idx, 0] - psi_true[-npred:][idx, 0] ,
          cal_residual[idx]
          ]

titles = [
          r'$\psi - \tilde \psi$',
          r'$ D(\psi)$'
          ]

subplots_2d(values, titles)
# %%
def plot_grid(psi, qhat):
    levels=10

    plt.contour(
        R[1:-1], Z[1:-1], psi, colors="blue", alpha=0.4,
        levels=levels
    )

    plt.contour(
    R[1:-1], Z[1:-1], qhat, colors="red", alpha=0.4,
    levels=levels
    )

    # plt.contour(
    # R[1:-1], Z[1:-1], -qhat, colors="green", alpha=0.4,
    # levels=levels
    # )

    plt.plot(
        np.array(df_jet_wall)[:, 0],
        np.array(df_jet_wall)[:, 1],
        linewidth=4,
        color="black",
        label="tokamak",
    )
    plt.axis("equal")
    plt.xlabel("R")
    plt.ylabel("Z")
    plt.title('Residual and Error Bar')

    plt.show()

idx = 10
qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=0.5)
plot_grid(pred_residual[idx], qhat)


# %%

alpha = 0.1
qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
prediction_sets = [- qhat,  + qhat]
Z_pos = 16
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

idx = 20
t_idx = -1

plt.plot(R[1:-1], pred_residual[idx][:, Z_pos], label='PRE', color='black',lw=4, ls='--', alpha=0.75)
plt.plot(R[1:-1], prediction_sets[0][:, Z_pos], label='Lower Marginal', color='maroon',lw=4, ls='--',  alpha=0.75)
plt.plot(R[1:-1], prediction_sets[1][:, Z_pos], label='Upper Marginal', color='red',lw=4, ls='--',  alpha=0.75)

plt.xlabel(r'$R$', fontsize=36)
plt.ylabel(r'$D(u)$', fontsize=36)

# %% 