# Calibrated Physics-Informed Uncertainty Quantification

A framework for providing calibrated, physics-informed uncertainty estimates for neural PDE solvers using conformal prediction. This approach leverages physics residual errors as a nonconformity score within a conformal prediction framework to enable data-free, model-agnostic, and statistically guaranteed uncertainty estimates.

## Key Features

- Physics Residual Error (PRE) as a nonconformity score for Conformal Prediction
- Data-free uncertainty quantification
- Model-agnostic implementation
- Marginal and Joint coverage guarantees
- Efficient gradient estimation using convolutional kernels

## Repository Structure

```
├── Active_Learning/        # Active learning experiments
├── Expts_initial/         # Initial experiments
├── Joint/                 # Joint conformal prediction implementation
├── Marginal/             # Marginal conformal prediction implementation
├── Neural_PDE/           # Neural PDE solver implementations
├── Physics_Informed/     # Physics-informed components
├── Tests/                # Test suite
├── Utils/                # Utility functions
└── __pycache__/         # Python cache files
```

## Experiments

The repository includes implementations for the following PDEs:

1. 1D Advection Equation
   - Domain: x ∈ [0, 2], t ∈ [0, 0.5]
   - Parameterized initial conditions

2. 1D Burgers' Equation
   - Domain: x ∈ [0, 2], t ∈ [0, 1.25]
   - Kinematic viscosity ν = 0.002

3. 2D Wave Equation
   - Domain: x, y ∈ [−1, 1], t ∈ [0, 1.0]
   - Wave speed c = 1.0

4. 2D Navier-Stokes Equations
   - Domain: x, y ∈ [0, 1], t ∈ [0, 0.5]
   - Incompressible fluid flow

5. 2D Magnetohydrodynamics (MHD)
   - Domain: x, y ∈ [0, 1]², t ∈ [0, 5]
   - Ideal MHD equations

## Requirements

- PyTorch
- NumPy
- SciPy
- tqdm

## Usage

### Basic Example

```python
from ConvOps_2d import ConvOperator

# Define operators for PDE
D_t = ConvOperator(domain='t', order=1)  # time-derivative
D_xx_yy = ConvOperator(domain=('x','y'), order=2)  # Laplacian
D_identity = ConvOperator()  # Identity Operator

# Combine operators
alpha, beta = 1.0, 0.5
D = ConvOperator()
D.kernel = D_t.kernel - alpha * D_xx_yy.kernel - beta * D_identity.kernel

# Estimate PRE
y_pred = model(X)
PRE = D(y_pred)
```

### Running Experiments

Each experiment directory contains specific implementation details and can be run independently:

```bash
python -m Expts_initial.advection_1d  # Run 1D advection experiment
python -m Expts_initial.burgers_1d    # Run 1D Burgers experiment
python -m Expts_initial.wave_2d       # Run 2D wave experiment
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{gopakumar2025calibrated,
  title={Calibrated Physics-Informed Uncertainty Quantification},
  author={Gopakumar, Vignesh and Giles, Dan and Kusner, Matt J. and Deisenroth, Marc Peter and Gray, Ander and Zanisi, Lorenzo and Pamela, Stanislas},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```

## License

[Add License Information]

## Contributors

- Vignesh Gopakumar
- Dan Giles
- Matt J. Kusner
- Marc Peter Deisenroth
- Ander Gray
- Lorenzo Zanisi
- Stanislas Pamela

## Contact

For questions and feedback, please contact v.gopakumar@ucl.ac.uk
