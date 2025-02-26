'''
Simple Harmonic Oscillator:

Eqn:
     m(d²x/dt²) + kx = 0

First order system:
    dx/dt = v
    dv/dt = -(k/m)x

    
'''

# %%
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append("/Users/Vicky/Documents/UKAEA/Code/Uncertainty_Quantification/PDE_Residuals")
from Utils import ConvOps_1d
# %% 
from scipy.integrate import solve_ivp

class HarmonicOscillator:
    """Numerical simple harmonic oscillator implementation."""
    
    def __init__(self, k=1.0, m=1.0):
        """
        Initialize the harmonic oscillator.
        
        Args:
            k (float): Spring constant
            m (float): Mass
        """
        self.k = k
        self.m = m
        self.omega = np.sqrt(k/m)
    
    def get_state_derivative(self, t, state):
        """
        Compute the derivative of the state vector [x, v].
        
        Args:
            t (float): Time (unused for autonomous system)
            state (np.ndarray): State vector [x, v]
            
        Returns:
            np.ndarray: Derivative of state vector [dx/dt, dv/dt]
        """
        x, v = state
        dx_dt = v
        dv_dt = -(self.k/self.m) * x
        return np.array([dx_dt, dv_dt])
    
    def solve_ode(self, t_span, initial_state, t_eval=None):
        """
        Solve the ODE numerically using scipy.integrate.solve_ivp
        
        Args:
            t_span (tuple): Time span (t_start, t_end)
            initial_state (np.ndarray): Initial state [x0, v0]
            t_eval (np.ndarray, optional): Times at which to evaluate solution
            
        Returns:
            tuple: Time points and solution array
        """
        solution = solve_ivp(
            fun=self.get_state_derivative,
            t_span=t_span,
            y0=initial_state,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-8
        )
        return solution.t, solution.y.T
    

#Neural network for the Neural ODE
class ODEFunc(nn.Module):
    """Neural network representing the ODE function."""
    
    def __init__(self, hidden_dim):
        """
        Initialize the neural ODE function.
        
        Args:
            hidden_dim (int): Number of hidden units
        """
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )
        
    def forward(self, t, y):
        """
        Forward pass of the neural network.
        
        Args:
            t (torch.Tensor): Time point (unused for autonomous system)
            y (torch.Tensor): State vector
            
        Returns:
            torch.Tensor: Predicted derivative
        """
        return self.net(y)

def generate_training_data(oscillator, t_span, n_points, n_trajectories):
    """
    Generate training data using numerical integration.
    
    Args:
        oscillator (HarmonicOscillator): Oscillator instance
        t_span (tuple): Time span (t_start, t_end)
        n_points (int): Number of points per trajectory
        n_trajectories (int): Number of trajectories
        
    Returns:
        tuple: Times, states, and derivatives arrays
    """
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    states = []
    derivatives = []
    
    for _ in range(n_trajectories):
        # Random initial conditions
        x0 = np.random.uniform(-2, 2)
        v0 = np.random.uniform(-2, 2)
        initial_state = np.array([x0, v0])
        
        # Get numerical solution
        _, solution = oscillator.solve_ode(t_span, initial_state, t_eval)
        states.append(solution)
        
        # Compute derivatives
        derivs = np.array([oscillator.get_state_derivative(_, state) 
                          for state in solution])
        derivatives.append(derivs)
    
    return (t_eval, 
            np.stack(states, axis=0),
            np.stack(derivatives, axis=0))

def train_neural_ode(func, train_t, train_states, train_derivs, n_epochs, batch_size):
    """
    Train the neural ODE.
    
    Args:
        func (ODEFunc): Neural network instance
        train_t (np.ndarray): Time points
        train_states (np.ndarray): State trajectories
        train_derivs (np.ndarray): State derivatives
        n_epochs (int): Number of training epochs
        batch_size (int): Batch size
        
    Returns:
        list: Training losses
    """
    optimizer = torch.optim.Adam(func.parameters())
    losses = []
    
    # Convert to PyTorch tensors
    train_states = torch.FloatTensor(train_states)
    train_derivs = torch.FloatTensor(train_derivs)
    
    n_samples = train_states.shape[0]
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        
        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            batch_states = train_states[i:i+batch_size]
            batch_derivs = train_derivs[i:i+batch_size]
            
            pred_derivs = func(0, batch_states.reshape(-1, 2))
            loss = torch.mean((pred_derivs - batch_derivs.reshape(-1, 2))**2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        losses.append(epoch_loss)
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}, Loss: {epoch_loss:.6f}')
    
    return losses

def compare_solutions(oscillator, neural_ode, t_span, initial_state):
    """
    Compare numerical and neural ODE solutions.
    
    Args:
        oscillator (HarmonicOscillator): Oscillator instance
        neural_ode (ODEFunc): Trained neural network
        t_span (tuple): Time span (t_start, t_end)
        initial_state (np.ndarray): Initial state [x0, v0]
        
    Returns:
        tuple: Time points and solutions (numerical and neural)
    """
    t = torch.linspace(t_span[0], t_span[1], 100)
    
    # Numerical solution
    _, numerical_solution = oscillator.solve_ode(
        t_span, initial_state, t.numpy())
    
    # Neural ODE solution
    state_0 = torch.FloatTensor(initial_state)
    neural_solution = odeint(neural_ode, state_0, t)
    
    return (t.numpy(), 
            numerical_solution,
            neural_solution.detach().numpy())

def evaluate(oscillator, neural_ode, t_span, n_points, x_range, v_range, n_solves):
    """
    Compare numerical and neural ODE solutions.
    
    Args:
        oscillator (HarmonicOscillator): Oscillator instance
        neural_ode (ODEFunc): Trained neural network
        t_span (tuple): Time span (t_start, t_end)
        n_points: spatial points
        x_range: domain
        v_range: domain
        n_solves: size of the dataset
        initial_state (np.ndarray): Initial state x0, v0]
        
    Returns:
        tuple: Time points and solutions (numerical and neural)
    """
    t = torch.linspace(t_span[0], t_span[1], 100)
    
    num_solns = []
    neural_solns = []

    for ii in tqdm(range(n_solves)):
        
        x0 = np.random.uniform(*x_range)
        v0 = np.random.uniform(*v_range)
        initial_state = np.array([x0, v0])

        # Numerical solution
        _, numerical_solution = oscillator.solve_ode(
            t_span, initial_state, t.numpy())
        
        # Neural ODE solution
        state_0 = torch.FloatTensor(initial_state)
        neural_solution = odeint(neural_ode, state_0, t)

        num_solns.append(numerical_solution)
        neural_solns.append(neural_solution.detach().numpy())
    
    return (t.numpy(), 
            np.asarray(num_solns),
            np.asarray(neural_solns))


def plot_comparison(t, numerical_sol, neural_sol):
    """
    Plot comparison between numerical and neural ODE solutions.
    
    Args:
        t (np.ndarray): Time points
        numerical_sol (np.ndarray): Analytical solution
        neural_sol (np.ndarray): Neural ODE solution
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Position plot
    ax1.plot(t, numerical_sol[:, 0], 'b-', label='Analytical')
    ax1.plot(t, neural_sol[:, 0], 'r--', label='Neural ODE')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Position')
    ax1.legend()
    ax1.grid(True)
    
    # Velocity plot
    ax2.plot(t, numerical_sol[:, 1], 'b-', label='Analytical')
    ax2.plot(t, neural_sol[:, 1], 'r--', label='Neural ODE')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Velocity')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# %% 
# Initialize system
m, k = 1.0, 1.0 
oscillator = HarmonicOscillator(k, m)

# Generate training data
t_span = (0, 10)
n_points = 100
n_trajectories = 50
t, states, derivs = generate_training_data(
    oscillator, t_span, n_points, n_trajectories)

# Initialize and train neural ODE
func = ODEFunc(hidden_dim=64)
losses = train_neural_ode(
    func, t, states, derivs, n_epochs=1000, batch_size=16)

# Compare solutions
initial_state = np.array([1.0, 0.0])  # x0 = 1, v0 = 0
t, numerical_sol, neural_sol = compare_solutions(
    oscillator, func, t_span, initial_state)

# Plot results
plot_comparison(t, numerical_sol, neural_sol)

# %% 
#PRE Estimations

soln = neural_sol
x = torch.tensor(soln[:, 0], dtype=torch.float32).unsqueeze(0)
v = torch.tensor(soln[:, 1], dtype=torch.float32).unsqueeze(0)


from Utils.ConvOps_0d import ConvOperator
dt = t[1]-t[0]
D_t = ConvOperator(order=1)#, scale=alpha)
D_tt = ConvOperator(order=2)#, scale=alpha)

D_identity = ConvOperator(order=0) #Identity 
D_identity.kernel = torch.tensor([0, 1, 0])

D_pos = ConvOperator(conv='direct')
D_pos.kernel = m*D_tt.kernel + dt**2*k*D_identity.kernel

D_pos_spectral = ConvOperator(conv='spectral')
D_pos_spectral.kernel = m*D_tt.kernel + dt**2*k*D_identity.kernel

# %% 
#Position Residual 
plt.figure()
plt.plot(t[1:-1], D_pos(x)[0,1:-1], 'b-', label='direct_residual')
plt.plot(t[1:-1], D_pos_spectral(x)[0,1:-1], 'r--', label='spectral_residual')
plt.plot(t[1:-1], D_pos.differentiate(x, correlation=True)[0, 1:-1], 'k:', label='custom_spectral')

plt.xlabel('Time')
plt.ylabel('Residual')
plt.title('Position Residual')
plt.legend()
plt.grid(True)


# #Velocity Residual 
# plt.figure()
# plt.plot(t[1:-1], D_vel(v)[0,1:-1], 'b-', label='velocity_residual')
# plt.xlabel('Time')
# plt.ylabel('Residual')
# plt.title('Velocity Residual')
# plt.legend()
# plt.grid(True)


# %%
#Performing PRE-CP
from Neural_PDE.UQ.inductive_cp import * 

t, numerical_sol, neural_sol = evaluate(
    oscillator, func, t_span, n_points, x_range=(-2,2), v_range=(-2,2), n_solves=5)

np.save("ODE_outputs", numerical_sol)
np.save("Nueral_outputs", numerical_sol)

pos = torch.tensor(neural_sol[...,0], dtype=torch.float32)
res = D_pos(pos)
residual_cal = res[:400]
residual_pred = res[400:]

# ncf_scores = np.abs(residual_cal) 

# #Emprical Coverage for all values of alpha to see if pred_residual lies between +- qhat. 
# alpha_levels = np.arange(0.05, 0.95+0.1, 0.1)
# emp_cov_res = []
# for alpha in tqdm(alpha_levels):
#     qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=alpha)
#     prediction_sets = [- qhat, + qhat]
#     emp_cov_res.append(emp_cov(prediction_sets, residual_pred.numpy()))

# plt.figure()
# plt.plot(1-alpha_levels, 1-alpha_levels, label='Ideal', color ='black', alpha=0.8, linewidth=3.0)
# plt.plot(1-alpha_levels, emp_cov_res, label='Residual' ,ls='-.', color='teal', alpha=0.8, linewidth=3.0)
# plt.xlabel('1-alpha')
# plt.ylabel('Empirical Coverage')
# plt.legend()

# #Plotting the bounds in the residual space
# qhat = calibrate(scores=ncf_scores, n=len(ncf_scores), alpha=0.1)

# plt.plot(t[1:-1], residual_pred[0, 1:-1], 'b-', label='PRE')
# plt.plot(t[1:-1], -qhat[1:-1], 'r--', label='Lower Bound')
# plt.plot(t[1:-1], +qhat[1:-1], 'g--', label='Upper Bound')
# plt.xlabel('Time')
# plt.ylabel('Residual')
# plt.title('PRE-CP : Position')
# plt.legend()
# plt.grid(True)

# %% 
#Inverse - Position : To perform the inverse you need the complete information. 
pos_res = D_pos.differentiate(torch.tensor(neural_sol[...,0], dtype=torch.float32), correlation=False, slice_pad=False)
pos_retrieved = D_pos.integrate(pos_res, correlation=False, slice_pad=False)

plt.plot(neural_sol[0, :, 0], 'b-', label='Actual')
plt.plot(pos_retrieved[0,1:-1], 'r--', label='Retrieved')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Inverse')
plt.legend()
plt.grid(True)

# %%
# # Attempting to invert the residual bounds to the velocity space just using the convolution theorem
# inv_qhat= D_pos.integrate(torch.tensor(qhat, dtype=torch.float32).unsqueeze(0))[0]

# plt.plot(t[1:-1], pos[0, 1:-1], 'b-', label='PRE')
# plt.plot(t[1:-1], -inv_qhat[1:-1], 'r--', label='Lower Bound')
# plt.plot(t[1:-1], +inv_qhat[1:-1], 'g--', label='Upper Bound')
# plt.xlabel('Time')
# plt.ylabel('Residual')
# plt.title('PRE-CP-inverse : Position')
# plt.legend()
# plt.grid(True)


# %%
