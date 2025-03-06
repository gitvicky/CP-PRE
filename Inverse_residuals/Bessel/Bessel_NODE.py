'''
Bessel Equation:

The Bessel equation of order n is:
    x^2(d²y/dx²) + x(dy/dx) + (x^2 - n^2)y = 0

For a first-order system, we can introduce v = dy/dx:
    dy/dx = v
    dv/dx = -v/x - (1 - n^2/x^2)y

Note: The Bessel equation is singular at x=0, requiring special handling.
'''

# %% 
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import solve_ivp
from scipy.special import jv, jvp  # Bessel functions and derivatives

import sys
sys.path.append("/Users/Vicky/Documents/UKAEA/Code/Uncertainty_Quantification/PDE_Residuals")
from Utils.ConvOps_0d import ConvOperator


class BesselEquation:
    """Numerical Bessel equation implementation."""
    
    def __init__(self, n=0):
        """
        Initialize the Bessel equation solver.
        
        Args:
            n (float): Order of the Bessel equation
        """
        self.n = n
        
    def get_state_derivative(self, x, state):
        """
        Compute the derivative of the state vector [y, v].
        
        Args:
            x (float): Independent variable
            state (np.ndarray): State vector [y, v]
            
        Returns:
            np.ndarray: Derivative of state vector [dy/dx, dv/dx]
        """
        y, v = state
        
        # Handle singularity at x=0
        if abs(x) < 1e-10:
            # For x->0, if n=0: v'->0, if n=1: v'->-y/2, if n>=2: v'->-infinity
            if self.n == 0:
                dv_dx = 0
            elif self.n == 1:
                dv_dx = -y/2
            else:
                # Approximate using small x value
                x = 1e-10
                dv_dx = -v/x - (1 - (self.n**2)/(x**2)) * y
        else:
            dv_dx = -v/x - (1 - (self.n**2)/(x**2)) * y
            
        dy_dx = v
        return np.array([dy_dx, dv_dx])
    
    def solve_ode(self, x_span, initial_state, x_eval=None):
        """
        Solve the ODE numerically using scipy.integrate.solve_ivp
        
        Args:
            x_span (tuple): Domain span (x_start, x_end)
            initial_state (np.ndarray): Initial state [y0, v0]
            x_eval (np.ndarray, optional): Points at which to evaluate solution
            
        Returns:
            tuple: x points and solution array
        """
        solution = solve_ivp(
            fun=self.get_state_derivative,
            t_span=x_span,
            y0=initial_state,
            t_eval=x_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-8
        )
        return solution.t, solution.y.T
    
    def analytical_solution(self, x):
        """
        Compute the analytical solution (Bessel function of the first kind)
        
        Args:
            x (np.ndarray): Points at which to evaluate
            
        Returns:
            tuple: y and dy/dx values
        """
        y = jv(self.n, x)
        dy_dx = jvp(self.n, x, 1)  # First derivative
        return y, dy_dx


# Neural network for the Neural ODE
class ODEFunc(nn.Module):
    """Neural network representing the ODE function."""
    
    def __init__(self, hidden_dim, n=0):
        """
        Initialize the neural ODE function.
        
        Args:
            hidden_dim (int): Number of hidden units
            n (float): Order of the Bessel equation
        """
        super(ODEFunc, self).__init__()
        self.n = n
        
        # Main network
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),  # Input: [y, v, x]
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)   # Output: [dy/dx, dv/dx]
        )
        
    def forward(self, x, y):
        """
        Forward pass of the neural network.
        
        Args:
            x (torch.Tensor): Independent variable
            y (torch.Tensor): State vector
            
        Returns:
            torch.Tensor: Predicted derivative
        """
        # Avoid singularity at x=0
        x_safe = x.clone()
        x_safe[x_safe < 1e-6] = 1e-6
        
        # Concatenate state and independent variable
        state_x = torch.cat([y, x_safe.reshape(-1, 1)], dim=1)
        return self.net(state_x)


def generate_training_data(bessel_equation, x_span, n_points, n_trajectories):
    """
    Generate training data using numerical integration.
    
    Args:
        bessel_equation (BesselEquation): Bessel equation instance
        x_span (tuple): Domain span (x_start, x_end)
        n_points (int): Number of points per trajectory
        n_trajectories (int): Number of trajectories
        
    Returns:
        tuple: x points, states, and derivatives arrays
    """
    x_eval = np.linspace(x_span[0], x_span[1], n_points)
    states = []
    derivatives = []
    
    for _ in range(n_trajectories):
        # Different initial conditions based on phase shift
        phase = np.random.uniform(0, 2*np.pi)
        
        # Start at x_span[0] with initial conditions from phase-shifted Bessel function
        n = bessel_equation.n
        x0 = x_span[0]
        
        if x0 < 1e-6:
            # Special case for x=0
            if n == 0:
                y0 = 1.0
                v0 = 0.0
            elif n == 1:
                y0 = 0.0
                v0 = 0.5
            else:
                y0 = 0.0
                v0 = 0.0
        else:
            # Use analytical solution with phase shift
            y0 = np.cos(phase) * jv(n, x0)
            v0 = np.cos(phase) * jvp(n, x0, 1)
            
        initial_state = np.array([y0, v0])
        
        # Get numerical solution
        _, solution = bessel_equation.solve_ode(x_span, initial_state, x_eval)
        states.append(solution)
        
        # Compute derivatives at each point
        derivs = np.array([bessel_equation.get_state_derivative(x, state) 
                          for x, state in zip(x_eval, solution)])
        derivatives.append(derivs)
    
    return (x_eval, 
            np.stack(states, axis=0),
            np.stack(derivatives, axis=0))


def train_neural_ode(func, train_x, train_states, train_derivs, n_epochs, batch_size):
    """
    Train the neural ODE.
    
    Args:
        func (ODEFunc): Neural network instance
        train_x (np.ndarray): x points
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
    train_x = torch.FloatTensor(train_x)
    train_states = torch.FloatTensor(train_states)
    train_derivs = torch.FloatTensor(train_derivs)
    
    n_samples = train_states.shape[0] * train_states.shape[1]
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        
        # Reshape data for batch processing
        x_flat = train_x.repeat(train_states.shape[0])
        states_flat = train_states.reshape(-1, 2)
        derivs_flat = train_derivs.reshape(-1, 2)
        
        # Create indices for shuffling
        indices = torch.randperm(n_samples)
        x_flat = x_flat[indices]
        states_flat = states_flat[indices]
        derivs_flat = derivs_flat[indices]
        
        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            batch_x = x_flat[i:i+batch_size].unsqueeze(1)
            batch_states = states_flat[i:i+batch_size]
            batch_derivs = derivs_flat[i:i+batch_size]
            
            pred_derivs = func(batch_x, batch_states)
            loss = torch.mean((pred_derivs - batch_derivs)**2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_states.shape[0]
        
        epoch_loss /= n_samples
        losses.append(epoch_loss)
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}, Loss: {epoch_loss:.6f}')
    
    return losses


def vectorized_ode_func(t, y, neural_net):
    """
    Vectorized ODE function for use with solve_ivp.
    
    Args:
        t (float): Independent variable
        y (np.ndarray): State vector
        neural_net (ODEFunc): Neural network instance
        
    Returns:
        np.ndarray: Derivative of state vector
    """
    y_tensor = torch.FloatTensor(y).reshape(1, -1)
    t_tensor = torch.FloatTensor([t]).reshape(1, 1)
    
    with torch.no_grad():
        dy = neural_net(t_tensor, y_tensor).numpy().flatten()
        
    return dy


def compare_solutions(bessel_equation, neural_ode, x_span, initial_state):
    """
    Compare numerical and neural ODE solutions.
    
    Args:
        bessel_equation (BesselEquation): Bessel equation instance
        neural_ode (ODEFunc): Trained neural network
        x_span (tuple): Domain span (x_start, x_end)
        initial_state (np.ndarray): Initial state [y0, v0]
        
    Returns:
        tuple: x points, solutions (numerical, analytical, and neural)
    """
    x_eval = np.linspace(x_span[0], x_span[1], 100)
    
    # Numerical solution (RK45)
    _, numerical_solution = bessel_equation.solve_ode(
        x_span, initial_state, x_eval)
    
    # Analytical solution (built-in Bessel function)
    analytical_y, analytical_v = bessel_equation.analytical_solution(x_eval)
    analytical_solution = np.column_stack((analytical_y, analytical_v))
    
    # Neural ODE solution
    state_0 = torch.FloatTensor(initial_state)
    x_tensor = torch.FloatTensor(x_eval)
    neural_solution = odeint(lambda t, y: neural_ode(t.reshape(-1, 1), y.reshape(-1, 2)).reshape(-1),
                           state_0, x_tensor)
    
    return (x_eval, 
            numerical_solution,
            analytical_solution,
            neural_solution.detach().numpy())


def evaluate(bessel_equation, neural_ode, x_span, n_points, n_solves):
    """
    Compare numerical and neural ODE solutions for multiple initial conditions.
    
    Args:
        bessel_equation (BesselEquation): Bessel equation instance
        neural_ode (ODEFunc): Trained neural network
        x_span (tuple): Domain span (x_start, x_end)
        n_points (int): Number of evaluation points
        n_solves (int): Number of different initial conditions
        
    Returns:
        tuple: x points and solutions (numerical, analytical, and neural)
    """
    x_eval = np.linspace(x_span[0], x_span[1], n_points)
    x_tensor = torch.FloatTensor(x_eval)
    
    num_solns = []
    analytical_solns = []
    neural_solns = []

    for _ in tqdm(range(n_solves)):
        # Different initial conditions based on phase shift
        phase = np.random.uniform(0, 2*np.pi)
        n = bessel_equation.n
        x0 = x_span[0]
        
        if x0 < 1e-6:
            # Special case for x=0
            if n == 0:
                y0 = np.random.uniform(0.5, 1.5)
                v0 = 0.0
            elif n == 1:
                y0 = 0.0
                v0 = np.random.uniform(0.25, 0.75)
            else:
                y0 = 0.0
                v0 = 0.0
        else:
            # Use analytical solution with phase shift and amplitude
            amplitude = np.random.uniform(0.5, 1.5)
            y0 = amplitude * np.cos(phase) * jv(n, x0)
            v0 = amplitude * np.cos(phase) * jvp(n, x0, 1)
            
        initial_state = np.array([y0, v0])
        
        # Numerical solution
        _, numerical_solution = bessel_equation.solve_ode(
            x_span, initial_state, x_eval)
        
        # Analytical solution (built-in Bessel function)
        if x0 < 1e-6 and n == 0 and abs(y0 - 1.0) < 1e-6 and abs(v0) < 1e-6:
            # This is exactly J_0(x)
            analytical_y = jv(n, x_eval)
            analytical_v = jvp(n, x_eval, 1)
        else:
            # Need to solve the IVP to get the right linear combination of Bessel functions
            analytical_y, analytical_v = zip(*[bessel_equation.analytical_solution(x) for x in x_eval])
            analytical_y = np.array(analytical_y)
            analytical_v = np.array(analytical_v)
            
            # Scale to match initial conditions
            if abs(analytical_y[0]) > 1e-6:
                scale = y0 / analytical_y[0]
                analytical_y *= scale
                analytical_v *= scale
        
        analytical_solution = np.column_stack((analytical_y, analytical_v))
        
        # Neural ODE solution
        state_0 = torch.FloatTensor(initial_state)
        neural_solution = odeint(lambda t, y: neural_ode(t.reshape(-1, 1), y.reshape(-1, 2)).reshape(-1),
                               state_0, x_tensor)

        num_solns.append(numerical_solution)
        analytical_solns.append(analytical_solution)
        neural_solns.append(neural_solution.detach().numpy())
    
    return (x_eval, 
            np.array(num_solns),
            np.array(analytical_solns),
            np.array(neural_solns))


def plot_comparison(x, numerical_sol, analytical_sol, neural_sol, n):
    """
    Plot comparison between numerical, analytical, and neural ODE solutions.
    
    Args:
        x (np.ndarray): x points
        numerical_sol (np.ndarray): Numerical solution
        analytical_sol (np.ndarray): Analytical solution
        neural_sol (np.ndarray): Neural ODE solution
        n (float): Order of the Bessel equation
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Function plot
    ax1.plot(x, numerical_sol[:, 0], 'b-', label='Numerical')
    ax1.plot(x, analytical_sol[:, 0], 'g--', label='Analytical (Jn)')
    ax1.plot(x, neural_sol[:, 0], 'r-.', label='Neural ODE')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title(f'Bessel Function J{n}(x)')
    
    # Derivative plot
    ax2.plot(x, numerical_sol[:, 1], 'b-', label='Numerical')
    ax2.plot(x, analytical_sol[:, 1], 'g--', label='Analytical (J\'n)')
    ax2.plot(x, neural_sol[:, 1], 'r-.', label='Neural ODE')
    ax2.set_xlabel('x')
    ax2.set_ylabel('dy/dx')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title(f'Derivative of Bessel Function J{n}(x)')
    
    plt.tight_layout()
    plt.show()


def plot_phase_space(numerical_sol, analytical_sol, neural_sol, n):
    """
    Plot phase space comparison between numerical, analytical, and neural ODE solutions.
    
    Args:
        numerical_sol (np.ndarray): Numerical solution
        analytical_sol (np.ndarray): Analytical solution
        neural_sol (np.ndarray): Neural ODE solution
        n (float): Order of the Bessel equation
    """
    plt.figure(figsize=(8, 6))
    plt.plot(numerical_sol[:, 0], numerical_sol[:, 1], 'b-', label='Numerical')
    plt.plot(analytical_sol[:, 0], analytical_sol[:, 1], 'g--', label='Analytical')
    plt.plot(neural_sol[:, 0], neural_sol[:, 1], 'r-.', label='Neural ODE')
    plt.xlabel('y')
    plt.ylabel('dy/dx')
    plt.title(f'Phase Space Trajectory (Bessel J{n})')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.show()


def analyze_residuals(x, neural_sol, n):
    """
    Analyze residuals using convolutional operators.
    
    Args:
        x (np.ndarray): x points
        neural_sol (np.ndarray): Neural ODE solution
        n (float): Order of the Bessel equation
        
    Returns:
        tuple: Residuals and retrieved solution
    """
    dx = x[1] - x[0]
    
    # Extract function and derivative
    y = torch.tensor(neural_sol[:, 0], dtype=torch.float32).unsqueeze(0)
    v = torch.tensor(neural_sol[:, 1], dtype=torch.float32).unsqueeze(0)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    
    # Initialize operators
    D_x = ConvOperator(order=1)
    D_xx = ConvOperator(order=2)
    D_identity = ConvOperator(order=0)
    
    # Create Bessel operator: x^2*D_xx + x*D_x + (x^2 - n^2)*identity
    # This is a simplified version that depends on x
    D_bessel = ConvOperator(conv='spectral')
    
    # Apply the Bessel equation operator
    residuals = torch.zeros_like(y)
    for i in range(len(x)):
        if abs(x[i]) < 1e-6:
            # Skip the singularity at x=0
            continue
            
        # Here we're computing x^2*y'' + x*y' + (x^2 - n^2)*y manually
        # In a full implementation, the operator would be constructed more elegantly
        if i > 0 and i < len(x) - 1:
            # Second derivative (central difference)
            y_xx = (y[0, i+1] - 2*y[0, i] + y[0, i-1]) / (dx**2)
            
            # First derivative (central difference)
            y_x = (y[0, i+1] - y[0, i-1]) / (2*dx)
            
            # Bessel equation residual
            residuals[0, i] = (x[i]**2) * y_xx + x[i] * y_x + (x[i]**2 - n**2) * y[0, i]
    
    # For a full implementation, this would invert the operator to retrieve y from residuals
    # This is a simplified placeholder
    y_retrieved = y
    
    return residuals, y_retrieved


# Main execution
if __name__ == "__main__":
    # Bessel equation orders to test
    n_values = [0, 1, 2]
    
    for n in n_values:
        print(f"\n=== Testing Bessel Equation of Order {n} ===")
        bessel_eq = BesselEquation(n=n)
        
        # Domain: avoid x=0 for higher orders due to singularity
        x_start = 0.01 if n > 0 else 0.0
        x_span = (x_start, 15.0)
        
        # Generate training data
        n_points = 150
        n_trajectories = 30
        x, states, derivs = generate_training_data(
            bessel_eq, x_span, n_points, n_trajectories)
        
        # Initialize and train neural ODE
        func = ODEFunc(hidden_dim=64, n=n)
        losses = train_neural_ode(
            func, x, states, derivs, n_epochs=1000, batch_size=32)
        
        # Plot training loss
        plt.figure(figsize=(8, 4))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss (Bessel J{n})')
        plt.yscale('log')
        plt.grid(True)
        plt.show()
        
        # Compare solutions
        # Use standard initial condition for J_n(x)
        if n == 0:
            initial_state = np.array([1.0, 0.0])  # J_0(0) = 1, J_0'(0) = 0
        elif n == 1:
            initial_state = np.array([0.0, 0.5])  # J_1(0) = 0, J_1'(0) = 1/2
        else:
            # Higher orders all have J_n(0) = 0, J_n'(0) = 0, 
            # but need to start at x > 0 due to singularity
            initial_state = np.array([jv(n, x_start), jvp(n, x_start, 1)])
            
        x, numerical_sol, analytical_sol, neural_sol = compare_solutions(
            bessel_eq, func, x_span, initial_state)
        
        # Plot results
        plot_comparison(x, numerical_sol, analytical_sol, neural_sol, n)
        plot_phase_space(numerical_sol, analytical_sol, neural_sol, n)
        
        # Analyze residuals
        residuals, y_retrieved = analyze_residuals(x, neural_sol, n)
        
        # Plot residuals
        plt.figure(figsize=(10, 6))
        plt.plot(x[1:-1], residuals[0, 1:-1], 'b-', label='Residual')
        plt.xlabel('x')
        plt.ylabel('Residual')
        plt.title(f'Bessel Equation Residual (J{n})')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Evaluate on multiple initial conditions
        print("\nEvaluating on multiple initial conditions...")
        x, num_solns, analytical_solns, neural_solns = evaluate(
            bessel_eq, func, x_span, n_points=100, n_solves=5)
            
        # Save results
        np.save(f"Bessel_numerical_outputs_n{n}", num_solns)
        np.save(f"Bessel_analytical_outputs_n{n}", analytical_solns)
        np.save(f"Bessel_neural_outputs_n{n}", neural_solns)
        
        print(f"Completed analysis for Bessel J{n}")

# %% 