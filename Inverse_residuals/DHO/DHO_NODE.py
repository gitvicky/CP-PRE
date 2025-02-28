'''
Damped Harmonic Oscillator:

Eqn:
     m(d²x/dt²) + c(dx/dt) + kx = 0

First order system:
    dx/dt - v = 0 
    dv/dt = -(k/m)x - (c/m)v

'''
# %%
#  
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import solve_ivp

# Import your ConvOps module
import sys
sys.path.append("/Users/Vicky/Documents/UKAEA/Code/Uncertainty_Quantification/PDE_Residuals")
from Utils.ConvOps_0d import ConvOperator


class DampedHarmonicOscillator:
    """Numerical damped harmonic oscillator implementation."""
    
    def __init__(self, k=1.0, m=1.0, c=0.2):
        """
        Initialize the damped harmonic oscillator.
        
        Args:
            k (float): Spring constant
            m (float): Mass
            c (float): Damping coefficient
        """
        self.k = k
        self.m = m
        self.c = c
        self.omega_0 = np.sqrt(k/m)  # Undamped angular frequency
        self.zeta = c / (2 * np.sqrt(m * k))  # Damping ratio
        
        # Calculate derived parameters for characterizing the system
        self.omega_d = self.omega_0 * np.sqrt(1 - self.zeta**2) if self.zeta < 1 else 0  # Damped angular frequency
        
        # Determine the type of damping
        if self.zeta < 1:
            self.damping_type = "Underdamped"
        elif self.zeta == 1:
            self.damping_type = "Critically damped"
        else:
            self.damping_type = "Overdamped"
            
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
        dv_dt = -(self.k/self.m) * x - (self.c/self.m) * v
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
        oscillator (DampedHarmonicOscillator): Oscillator instance
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
        oscillator (DampedHarmonicOscillator): Oscillator instance
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
    Compare numerical and neural ODE solutions for multiple initial conditions.
    
    Args:
        oscillator (DampedHarmonicOscillator): Oscillator instance
        neural_ode (ODEFunc): Trained neural network
        t_span (tuple): Time span (t_start, t_end)
        n_points: Number of spatial points
        x_range: Domain for position
        v_range: Domain for velocity
        n_solves: Size of the dataset
        
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


def plot_comparison(t, numerical_sol, neural_sol, damping_type=""):
    """
    Plot comparison between numerical and neural ODE solutions.
    
    Args:
        t (np.ndarray): Time points
        numerical_sol (np.ndarray): Analytical solution
        neural_sol (np.ndarray): Neural ODE solution
        damping_type (str): Type of damping for the title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Position plot
    ax1.plot(t, numerical_sol[:, 0], 'b-', label='Numerical')
    ax1.plot(t, neural_sol[:, 0], 'r--', label='Neural ODE')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Position')
    ax1.legend()
    ax1.grid(True)
    if damping_type:
        ax1.set_title(f'Position vs Time ({damping_type})')
    else:
        ax1.set_title('Position vs Time')
    
    # Velocity plot
    ax2.plot(t, numerical_sol[:, 1], 'b-', label='Numerical')
    ax2.plot(t, neural_sol[:, 1], 'r--', label='Neural ODE')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Velocity')
    ax2.legend()
    ax2.grid(True)
    if damping_type:
        ax2.set_title(f'Velocity vs Time ({damping_type})')
    else:
        ax2.set_title('Velocity vs Time')
    
    plt.tight_layout()
    plt.show()


def plot_phase_space(numerical_sol, neural_sol, damping_type=""):
    """
    Plot phase space comparison between numerical and neural ODE solutions.
    
    Args:
        numerical_sol (np.ndarray): Analytical solution
        neural_sol (np.ndarray): Neural ODE solution
        damping_type (str): Type of damping for the title
    """
    plt.figure(figsize=(8, 6))
    plt.plot(numerical_sol[:, 0], numerical_sol[:, 1], 'b-', label='Numerical')
    plt.plot(neural_sol[:, 0], neural_sol[:, 1], 'r--', label='Neural ODE')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    if damping_type:
        plt.title(f'Phase Space Trajectory ({damping_type})')
    else:
        plt.title('Phase Space Trajectory')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.show()


def analyze_residuals(t, neural_sol, m, c, k):
    """
    Analyze residuals using convolutional operators.
    
    Args:
        t (np.ndarray): Time points
        neural_sol (np.ndarray): Neural ODE solution
        m (float): Mass
        c (float): Damping coefficient
        k (float): Spring constant
        
    Returns:
        tuple: Residuals and retrieved solution
    """
    dt = t[1] - t[0]
    
    # Extract position and velocity
    x = torch.tensor(neural_sol[:, 0], dtype=torch.float32).unsqueeze(0)
    v = torch.tensor(neural_sol[:, 1], dtype=torch.float32).unsqueeze(0)
    
    # Initialize operators
    D_t = ConvOperator(order=1)
    D_tt = ConvOperator(order=2)
    D_identity = ConvOperator(order=0)
    D_identity.kernel = torch.tensor([0, 1, 0])
    
    # Create damped operator: m*D_tt + c*D_t + k*identity
    D_damped = ConvOperator(conv='spectral')
    D_damped.kernel = 2*m*D_tt.kernel + dt*c*D_t.kernel + 2*dt**2*k*D_identity.kernel
    
    # Calculate residuals
    residuals = D_damped(x)
    
    # Retrieve position through integration
    pos_res = D_damped.differentiate(x, correlation=True, slice_pad=False)
    pos_retrieved = D_damped.integrate(pos_res, correlation=True, slice_pad=False)
    
    return residuals, pos_res, pos_retrieved


# Main execution
if __name__ == "__main__":
    # System parameters
    m, k = 1.0, 1.0 
    
    # Test different damping scenarios
    damping_cases = [
        {'c': 0.2, 'name': 'Underdamped (ζ < 1)'},
        # {'c': 2.0, 'name': 'Critical/Overdamped (ζ ≥ 1)'}
    ]
    
    for case in damping_cases:
        c = case['c']
        name = case['name']
        
        print(f"\n=== Testing {name} ===")
        oscillator = DampedHarmonicOscillator(k, m, c)
        print(f"Damping ratio ζ = {oscillator.zeta:.3f}, Classification: {oscillator.damping_type}")
        
        # Generate training data
        t_span = (0, 15)  # Extended to see the damping effects
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
        plot_comparison(t, numerical_sol, neural_sol, oscillator.damping_type)
        plot_phase_space(numerical_sol, neural_sol, oscillator.damping_type)
        
        # Analyze residuals
        residuals, pos_res, pos_retrieved = analyze_residuals(t, neural_sol, m, c, k)
        
        # Plot residuals
        plt.figure(figsize=(10, 6))
        plt.plot(t[1:-1], residuals[0, 1:-1], 'b-', label='Position Residual')
        plt.xlabel('Time')
        plt.ylabel('Residual')
        plt.title(f'Position Residual ({oscillator.damping_type})')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot retrieved vs actual position
        plt.figure(figsize=(10, 6))
        plt.plot(t, neural_sol[:, 0], 'b-', label='Actual')
        plt.plot(t, pos_retrieved[0, 1:-1], 'r--', label='Retrieved')
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.title(f'Position: Actual vs Retrieved ({oscillator.damping_type})')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Evaluate on multiple initial conditions
        print("\nEvaluating on multiple initial conditions...")
        t, num_solns, neural_solns = evaluate(
            oscillator, func, t_span, n_points, x_range=(-2, 2), v_range=(-2, 2), n_solves=5)
            
        # Save results
        np.save(f"DHO_numerical_outputs_{int(c*10)}", num_solns)
        np.save(f"DHO_neural_outputs_{int(c*10)}", neural_solns)
        
        print(f"Completed analysis for {name}")

# %% 
#further analysis 
dt = t[1] - t[0]
# Extract position and velocity
x = torch.tensor(neural_sol[:, 0], dtype=torch.float32).unsqueeze(0)
v = torch.tensor(neural_sol[:, 1], dtype=torch.float32).unsqueeze(0)

# Initialize operators
D_t = ConvOperator(order=1)
D_tt = ConvOperator(order=2)
D_identity = ConvOperator(order=0)
D_identity.kernel = torch.tensor([0.0, 1.0, 0.0])

# Create damped operator: m*D_tt + c*D_t + k*identity
D_damped = ConvOperator(conv='spectral')
D_damped.kernel = 2*m*D_tt.kernel + dt*c*D_t.kernel + 2*dt**2*k*D_identity.kernel

# Calculate residuals
residuals = D_damped(x)

# Retrieve position through integration
pos_res = D_damped.differentiate(x, correlation=True, slice_pad=False)
pos_retrieved = D_damped.integrate(pos_res, correlation=True, slice_pad=False)


#Position Residual 
plt.figure()
plt.plot(t[1:-1], D_damped.convolution(x)[0,1:-1], 'b-', label='direct_residual')
plt.plot(t[1:-1], D_damped(x)[0,1:-1], 'r--', label='spectral_residual')
plt.plot(t[1:-1], D_damped.differentiate(x, correlation=True)[0, 1:-1], 'k:', label='custom_spectral')

plt.xlabel('Time')
plt.ylabel('Residual')
plt.title('Position Residual')
plt.legend()
plt.grid(True)

# %%
#Exploring the multivariate use cases where you will split the residual based on which variable R = R1 + R2 
x = torch.tensor(neural_sol[:, 0], dtype=torch.float32).unsqueeze(0)
v = torch.tensor(neural_sol[:, 1], dtype=torch.float32).unsqueeze(0)

D_R1 = ConvOperator()#v
D_R1.kernel = m*D_t.kernel + 2*dt*c*D_identity.kernel

D_R2 = ConvOperator()#x
D_R2.kernel =  2*dt*k*D_identity.kernel

residuals_r1r2 = D_R1.differentiate(v, correlation=True) + D_R2.differentiate(x, correlation=True)

plt.figure()
plt.plot(t[1:-1], D_damped.differentiate(x, correlation=True)[0, 1:-1], 'k:', label='combined residuals')
plt.plot(t[1:-1], residuals_r1r2[0, 1:-1], 'r--', label='split residuals')
plt.xlabel('Time')
plt.ylabel('Residual')
plt.title('Split Residual')
plt.legend()
plt.grid(True)
# %%

plt.figure()
plt.plot(t[1:-1], D_t(x)[0, 1:-1], 'k:', label='x_deriv')
plt.plot(t[1:-1], dt*2*v[0, 1:-1], 'r--', label='velocity')
plt.xlabel('Time')
plt.ylabel('Residual')
plt.title('Extracting velocity from x')
plt.legend()
plt.grid(True)
# %%
#Velocity 
inverse_r1 =  D_R1.integrate(D_R1.differentiate(v, correlation=True, slice_pad=False), correlation=True, slice_pad=True)
plt.figure()
plt.plot(t[1:-1], v[0, 1:-1], 'k:', label='velocity')
plt.plot(t[1:-1], inverse_r1[0, 2:-2], 'r--', label='split inverse')
plt.xlabel('Time')
plt.ylabel('Residual')
plt.title('Split Inverse - velocity ')
plt.legend()
plt.grid(True)

#Position
inverse_r2 =  D_R2.integrate(D_R2.differentiate(x, correlation=True, slice_pad=False), correlation=True, slice_pad=True)
plt.figure()
plt.plot(t[1:-1], x[0, 1:-1], 'k:', label='position')
plt.plot(t[1:-1], inverse_r2[0, 2:-2], 'r--', label='split inverse')
plt.xlabel('Time')
plt.ylabel('Residual')
plt.title('Split Inverse - position')
plt.legend()
plt.grid(True)
# %%

D_R3 = ConvOperator()#x
D_R3.kernel = D_t.kernel 

D_R4 = ConvOperator()#v
D_R4.kernel =  2*dt*D_identity.kernel

residuals_r3r4 = - D_R4.differentiate(v, correlation=True) + D_R3.differentiate(x, correlation=True)

plt.figure()
plt.plot(t[1:-1], residuals_r3r4[0, 1:-1], 'r--', label='split residuals')
plt.xlabel('Time')
plt.ylabel('Residual')
plt.title('Split Residual - velocity')
plt.legend()
plt.grid(True)
# %%
#Velocity 
inverse_r1 =  D_R4.integrate(D_R4.differentiate(v, correlation=True, slice_pad=False), correlation=True, slice_pad=True)
plt.figure()
plt.plot(t[1:-1], v[0, 1:-1], 'k:', label='velocity')
plt.plot(t[1:-1], inverse_r1[0, 2:-2], 'r--', label='split inverse')
plt.xlabel('Time')
plt.ylabel('Residual')
plt.title('Split Inverse - velocity ')
plt.legend()
plt.grid(True)

#Position
inverse_r2 =  D_R4.integrate(D_R4.differentiate(x, correlation=True, slice_pad=False), correlation=True, slice_pad=True)
plt.figure()
plt.plot(t[1:-1], x[0, 1:-1], 'k:', label='position')
plt.plot(t[1:-1], inverse_r2[0, 2:-2], 'r--', label='split inverse')
plt.xlabel('Time')
plt.ylabel('Residual')
plt.title('Split Inverse - position')
plt.legend()
plt.grid(True)
# %%
