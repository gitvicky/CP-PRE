# %%
import numpy as np
import matplotlib.pyplot as plt
from zonopy import interval, zonotope
import sys

# Add the current directory to the path to import our modules
sys.path.append('.')

# Import our custom modules
from pre_set_prop import set_PRE

def generate_synthetic_data():
    """
    Generate synthetic data for demonstration purposes when real data is not available.
    
    Returns:
    - Numerical solution and neural solution arrays
    """
    print("Generating synthetic data for demonstration...")
    
    # Number of samples, time points, and features
    n_samples = 300
    n_timesteps = 50
    n_features = 1
    
    # Time vector
    t = np.linspace(0, 2*np.pi, n_timesteps)
    
    # Create numerical solution (ground truth)
    numerical_sol = np.zeros((n_samples, n_timesteps, n_features))
    for i in range(n_samples):
        # Random frequency and phase
        freq = 0.5 + np.random.rand() * 1.5
        phase = np.random.rand() * np.pi
        amplitude = 0.5 + np.random.rand()
        
        # True solution: damped sine wave
        numerical_sol[i, :, 0] = amplitude * np.sin(freq * t + phase) * np.exp(-0.1 * t)
    
    # Create neural solution (approximate)
    neural_sol = numerical_sol + np.random.normal(0, 0.05, numerical_sol.shape)
    
    return numerical_sol, neural_sol


def test_with_synthetic_data():
    """Test the PRE set propagation with synthetic data."""
    # Generate synthetic data

# Load data
    numerical_sol = np.load("ODE_outputs.npy")
    neural_sol = np.load("Nueral_outputs.npy")

    
    # Select a random sample
    idx = np.random.randint(0, len(numerical_sol))
    print(f"Selected sample {idx} for testing")
    
    neural_test = neural_sol[idx]
    numerical_test = numerical_sol[idx]
    
    # Compute set bounds
    print("Computing set bounds...")
    signal_bounds = set_PRE(neural_test)
    signal_bounds_back = signal_bounds[1:-1]
    
    # Check if solutions are within bounds
    is_numerical_in = all(numerical_test[i, 0] in signal_bounds_back[i] for i in range(len(numerical_test)))
    is_neural_in = all(neural_test[i, 0] in signal_bounds_back[i] for i in range(len(neural_test)))
    
    print(f"Numerical solution contained within bounds: {is_numerical_in}")
    print(f"Neural solution contained within bounds: {is_neural_in}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    t = np.arange(len(neural_test))
    
    # Plot solutions
    plt.plot(t, neural_test[:, 0], 'b-', label='Neural solution')
    plt.plot(t, numerical_test[:, 0], 'r--', label='Numerical solution')
    
    # Plot bounds
    upper_bounds = [float(interval_obj.sup) for interval_obj in signal_bounds_back]
    lower_bounds = [float(interval_obj.inf) for interval_obj in signal_bounds_back]
    plt.fill_between(t[1:], lower_bounds, upper_bounds, color='gray', alpha=0.3, label='PRE bounds')
    
    plt.title('PRE Set Bounds Demonstration with Synthetic Data')
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('synthetic_pre_bounds.png')
    plt.show()


if __name__ == "__main__":
    test_with_synthetic_data()

# %%