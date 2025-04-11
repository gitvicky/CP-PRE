# PRE Set Propagation in Python

This repository contains a Python implementation of Physics-Regularized Error (PRE) set propagation for ODE solutions. The code propagates sets through spectral convolutions using FFT and interval arithmetic.

## Overview

The implementation converts the Julia code from `PRE_set_prop.jl` and `intervalFFT.jl` to Python. It uses:

- NumPy and SciPy for numerical operations and FFT
- python-interval for interval arithmetic
- Custom implementation of zonotopes for set propagation
- Matplotlib for visualization

## Structure

- `zonotope.py`: Implementation of the zonotope data structure
- `intervalFFT.py`: Implementation of interval arithmetic with FFT operations
- `pre_set_prop.py`: Main implementation of the PRE set propagation algorithm 
- `example.py`: Example script demonstrating usage with synthetic data
- `requirements.txt`: Required Python packages

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### With Real Data

If you have the ODE numerical solutions and neural network solutions:

```python
import numpy as np
from pre_set_prop import set_PRE

# Load data
numerical_sol = np.load("ODE_outputs_poor.npy")
neural_sol = np.load("Nueral_outputs_poor.npy")

# Select a test case
id = 42  # Or any other index
neural_test = neural_sol[id, :, :]
numerical_test = numerical_sol[id, :, :]

# Compute bounds
signal_bounds = set_PRE(neural_test)
signal_bounds_back = signal_bounds[1:-1]  # Remove padding

# Check if solutions are within bounds
is_numerical_in = all(numerical_test[i, 0] in signal_bounds_back[i] for i in range(len(numerical_test)))
is_neural_in = all(neural_test[i, 0] in signal_bounds_back[i] for i in range(len(neural_test)))

print(f"Numerical solution contained within bounds: {is_numerical_in}")
print(f"Neural solution contained within bounds: {is_neural_in}")
```

### Using Example Script

For testing with synthetic data:

```bash
python example.py
```

This will generate synthetic data, compute set bounds, and visualize the results.

## Technical Details

### Set Representation

Sets are represented as zonotopes, which are efficient for linear transformations such as those in FFT operations. A zonotope is defined by a center and a set of generators.

### FFT Operations

The implementation leverages the linearity of FFT and uses zonotopes to track set propagation through FFT and inverse FFT operations.

### PRE Method

The Physics-Regularized Error method:
1. Takes a neural network solution to an ODE
2. Computes the residual using spectral convolutions with predefined kernels
3. Constructs a set in the residual space
4. Propagates this set back to the solution space using FFT operations

## Notes on Implementation

- The Python implementation provides equivalent functionality to the Julia version but with some adaptations to accommodate differences in language capabilities
- The zonotope implementation includes methods for managing computational complexity
- Interval arithmetic is handled using the python-interval package, which provides similar functionality to Julia's IntervalArithmetic

## Example Output

When running the example script, you should see a plot showing:
- The neural network solution
- The numerical (ground truth) solution
- The computed PRE bounds shown as a shaded region

The bounds should contain both solutions if the implementation is working correctly.
