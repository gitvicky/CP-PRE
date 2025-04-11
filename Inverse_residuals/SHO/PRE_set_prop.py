import numpy as np
from scipy.fft import fft, ifft
from interval import interval, imath
import matplotlib.pyplot as plt

# Parameters
m = 1
k = 1
D_tt_kernel = np.array([1, -2, 1])
dt = 0.1010101
D_identity = np.array([0, 1, 0])

# Define inverse kernel function
def compute_inverse(kernel_fft, eps=1e-16):
    return 1 / (kernel_fft + eps)

# Mock intervalFFT and inverse_intervalFFT functions
# Replace these with your actual implementations
def intervalFFT(signal_intervals):
    return np.array([complex(x.mid, x.width) for x in signal_intervals])

def inverse_intervalFFT(signal_intervals):
    return np.array([interval(x.real, x.imag) for x in signal_intervals])

def complex_prod(a, b):
    return a * b

# Load data
numerical_sol = np.load("ODE_outputs_poor.npy")
neural_sol = np.load("Nueral_outputs_poor.npy")

def set_PRE(neural_test):
    D_pos_kernel = m * D_tt_kernel + dt**2 * k * D_identity

    signal_padded = np.concatenate(([0], neural_test[:, 0], [0]))

    N_signal = signal_padded.shape[0]
    N_pad = N_signal - len(D_pos_kernel)
    kernel_pad = np.concatenate((D_pos_kernel, np.zeros(N_pad)))

    signal_fft = fft(signal_padded)
    kernel_fft = fft(kernel_pad)

    convolved = ifft(signal_fft * kernel_fft)
    inverse_kernel = compute_inverse(kernel_fft)

    convolved_noedges = convolved[3:-1]
    right_edges = convolved[:3]
    left_edges = convolved[-1]

    convolved_set_center = interval(-np.abs(np.real(convolved_noedges)), np.abs(np.real(convolved_noedges)))
    convolved_set_right = interval(*np.real(right_edges))
    convolved_set_left = interval(np.real(left_edges))

    convolved_set = np.concatenate((convolved_set_right, convolved_set_center, convolved_set_left))

    convolved_set_fft = intervalFFT(convolved_set)
    convolved_set_fft_kernel = complex_prod(convolved_set_fft, inverse_kernel)
    retrieved_signal = inverse_intervalFFT(convolved_set_fft_kernel)

    return np.array([x.mid for x in retrieved_signal])

# Test the pipeline
ID = np.random.randint(0, 300)
neural_test = neural_sol[ID, :, 0]
numerical_test = numerical_sol[ID, :, 0]

signal_bounds = set_PRE(neural_test)
signal_bounds_back = signal_bounds[1:-1]

is_it_in_numerical = np.all(np.isin(numerical_test, signal_bounds_back))
is_it_in_neural = np.all(np.isin(neural_test, signal_bounds_back))

print(f"Numerical is in: {is_it_in_numerical} and Neural is in: {is_it_in_neural}")

# Plotting
plt.plot(neural_test, label="neural")
plt.plot(numerical_test, label="numerical")
plt.fill_between(
    range(len(signal_bounds_back)),
    [x[0] for x in signal_bounds_back],
    [x[1] for x in signal_bounds_back],
    alpha=0.2,
    label="signal bounds"
)
plt.legend()
plt.show()