# %%
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import random

# Import our custom modules
from zonopy import zonotope, interval
from intervalFFT import (
    intervalFFT, inverse_intervalFFT, Real, 
    complex_prod
)


def compute_inverse(kernel_fft, eps=1e-16):
    """
    Compute the inverse of a kernel in frequency domain with regularization.
    
    Parameters:
    - kernel_fft: FFT of the kernel
    - eps: Small number to avoid division by zero
    
    Returns:
    - Inverse of the kernel
    """
    return 1.0 / (kernel_fft + eps)


def set_PRE(neural_test):
    """
    Main function to compute PRE (Physics-Regularized Error) set bounds.
    
    Parameters:
    - neural_test: Neural network solution vector
    
    Returns:
    - List of intervals representing bounds on the solution
    """
    # Parameters (same as in the Julia code)
    m = 1
    k = 1
    dt = 0.1010101
    D_tt_kernel = np.array([1, -2, 1])
    D_identity = np.array([0, 1, 0])
    
    # Compute positive kernel
    D_pos_kernel = m * D_tt_kernel + dt**2 * k * D_identity
    
    # Pad the signal
    signal_padded = np.concatenate(([0], neural_test[:, 0], [0]))
    
    # Determine paddings
    N_signal = len(signal_padded)
    N_pad = N_signal - len(D_pos_kernel)
    kernel_pad = np.concatenate((D_pos_kernel, np.zeros(N_pad)))
    print(kernel_pad)
    
    # Compute FFT
    signal_fft = fft(signal_padded)
    kernel_fft = fft(kernel_pad)
    
    # Convolve in frequency domain and compute inverse
    convolved = ifft(signal_fft * kernel_fft)
    inverse_kernel = compute_inverse(kernel_fft)
    
    # Split convolved signal
    convolved_noedges = convolved[4:-1]
    right_edges = convolved[1:4]
    left_edges = convolved[-1]
    
    # Create interval sets for different parts
    convolved_set_center = [interval(-abs(x.real), abs(x.real)) for x in convolved_noedges]
    convolved_set_right = [interval(x.real, x.real) for x in right_edges]
    convolved_set_left = [interval(left_edges.real, left_edges.real)]
    
    # Combine all parts
    convolved_set = convolved_set_right + convolved_set_center + convolved_set_left
    
    # Perform interval FFT
    convolved_set_fft = intervalFFT(convolved_set)
    
    # Multiply with inverse kernel
    convolved_set_fft_kernel = [complex_prod(z, c) for z, c in zip(convolved_set_fft, inverse_kernel)]
    # complex_prod(convolved_set_fft[0], inverse_kernel[0])
    
    # Perform inverse interval FFT
    retrieved_signal = inverse_intervalFFT(convolved_set_fft_kernel)
    
    # Extract real parts
    return [Real(z) for z in retrieved_signal]


def main():
    """Main function to run the PRE set propagation"""
    print("Loading data...")
    # Load data
    try:
        numerical_sol = np.load("ODE_outputs.npy")
        neural_sol = np.load("Nueral_outputs.npy")
        print("Data loaded successfully!")
    except FileNotFoundError:
        print("Error: Could not find input data files. Please ensure ODE_outputs_poor.npy and Nueral_outputs_poor.npy exist.")
        return
    
    # Select a random ID
    ID = random.randint(0, 5)
    print(f"Selected random ID: {ID}")
    
    # Extract test data
    neural_test = neural_sol[ID, :, :]
    numerical_test = numerical_sol[ID, :, :]
    
    print(f"Neural solution shape: {neural_test.shape}")
    print(f"Numerical solution shape: {numerical_test.shape}")
    
    # Compute bounds
    print("Computing set bounds using PRE method...")
    signal_bounds = set_PRE(neural_test)
    signal_bounds_back = signal_bounds[1:-1]
    
    # Check if solutions are within bounds
    is_it_in_numerical = all(numerical_test[i, 0] in signal_bounds_back[i] for i in range(len(numerical_test)))
    is_it_in_neural = all(neural_test[i, 0] in signal_bounds_back[i] for i in range(len(neural_test)))
    
    print(f"Numerical is in: {is_it_in_numerical} and Neural is in: {is_it_in_neural}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    # Plot neural solution
    plt.plot(neural_test[:, 0], label="neural")
    
    # Plot numerical solution
    plt.plot(numerical_test[:, 0], label="numerical")
    
    # Plot bounds
    upper_bounds = [float(interval_obj.sup) for interval_obj in signal_bounds_back]
    lower_bounds = [float(interval_obj.inf) for interval_obj in signal_bounds_back]
    plt.fill_between(range(len(signal_bounds_back)), lower_bounds, upper_bounds, alpha=0.2)
    
    plt.legend()
    plt.title("ODE Solutions with PRE Set Bounds")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.savefig("pre_set_bounds.png")
    plt.show()


if __name__ == "__main__":
    main()
