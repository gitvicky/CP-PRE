# %%
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from interval import interval, imath
import scipy.spatial as spatial

class Zonotope:
    """
    Implementation of a zonotope for set-based computations.
    A zonotope is defined by a center and a list of generators.
    """
    def __init__(self, center, generators):
        """
        Initialize a zonotope with center and generators.
        
        Parameters:
        - center: A numpy array representing the center of the zonotope
        - generators: A numpy array where each column is a generator
        """
        self.center = np.array(center, dtype=float)
        self.generators = np.array(generators, dtype=float)
        self.dim = len(center)
    
    def vertices(self):
        """
        Compute the vertices of the zonotope using the convex hull method.
        
        Returns:
        - A numpy array where each row is a vertex
        """
        # Generate all possible combinations of generator coefficients [-1, 1]
        n_generators = self.generators.shape[1]
        coeffs = np.array(np.meshgrid(*[[-1, 1] for _ in range(n_generators)]))
        coeffs = coeffs.T.reshape(-1, n_generators)
        
        # Calculate all potential vertices
        potential_vertices = self.center + np.dot(coeffs, self.generators.T)
        
        # Use scipy's ConvexHull to find the vertices
        if self.dim <= 1 or n_generators <= 1:
            return potential_vertices
        else:
            try:
                hull = spatial.ConvexHull(potential_vertices)
                return potential_vertices[hull.vertices]
            except (spatial.QhullError, ValueError):
                # Fallback in case of degenerate cases
                return potential_vertices
    
    def __add__(self, other):
        """Minkowski sum of zonotopes"""
        if isinstance(other, Zonotope):
            center = self.center + other.center
            generators = np.hstack((self.generators, other.generators))
            return Zonotope(center, generators)
        else:
            raise TypeError("Addition is only defined for Zonotope objects")
    
    def __mul__(self, scalar):
        """Scale the zonotope by a scalar"""
        return Zonotope(scalar * self.center, scalar * self.generators)
    
    def __rmul__(self, scalar):
        """Right multiplication by scalar"""
        return self.__mul__(scalar)
    
    def linear_map(self, matrix):
        """Apply a linear map to the zonotope"""
        matrix = np.array(matrix)
        center = matrix @ self.center
        generators = matrix @ self.generators
        return Zonotope(center, generators)
    
    def high(self):
        """Get upper bounds for each dimension"""
        return self.center + np.sum(np.abs(self.generators), axis=1)
    
    def low(self):
        """Get lower bounds for each dimension"""
        return self.center - np.sum(np.abs(self.generators), axis=1)
    
    def contains(self, point):
        """Check if a point is contained within the zonotope"""
        point = np.array(point)
        shifted = point - self.center
        
        # Set up linear program to check if point is in zonotope
        # This is a simple implementation; more efficient algorithms exist
        if self.generators.shape[1] == 0:
            return np.allclose(shifted, 0)
        
        # Check using linear programming (approximate solution for demonstration)
        try:
            from scipy.optimize import linprog
            c = np.ones(self.generators.shape[1])
            A_eq = self.generators
            b_eq = shifted
            bounds = [(-1, 1) for _ in range(self.generators.shape[1])]
            result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
            return result.success and np.max(np.abs(result.x)) <= 1
        except:
            # Fallback to a simple approximation
            box_high = self.high()
            box_low = self.low()
            return np.all(point >= box_low) and np.all(point <= box_high)


def overapproximate_zonotope(zono):
    """
    Overapproximate a zonotope with another zonotope.
    In this simplified implementation, we just return the same zonotope.
    A real implementation might reduce the number of generators.
    """
    return zono


def convert_interval_to_zonotope(interval_obj):
    """
    Convert an interval to a zonotope.
    
    Parameters:
    - interval_obj: A Python interval object
    
    Returns:
    - A Zonotope object representing the interval
    """
    inf_val = float(interval_obj.inf)
    sup_val = float(interval_obj.sup)
    center = (inf_val + sup_val) / 2
    generator = (sup_val - inf_val) / 2
    return Zonotope([center, 0], [[generator, 0], [0, 0]])


def minkowski_sum(z1, z2):
    """Compute Minkowski sum of two zonotopes"""
    return z1 + z2


def scale(scalar, zonotope):
    """Scale a zonotope by a scalar"""
    return scalar * zonotope


def complex_prod(z_zonotope, complex_num):
    """
    Multiply a zonotopic complex number by a precise complex number.
    
    Parameters:
    - z_zonotope: A zonotope representing a complex number (real and imaginary parts)
    - complex_num: A complex number
    
    Returns:
    - The product as a zonotope
    """
    scaling_fac = abs(complex_num)
    angle = np.arctan2(complex_num.imag, complex_num.real)
    
    rot_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    z_rot = z_zonotope.linear_map(rot_matrix)
    return scale(scaling_fac, z_rot)


def intervalFFT_(Xk, h):
    """
    Compute a single component of the FFT of an interval vector.
    
    Parameters:
    - Xk: List of interval objects
    - h: Index
    
    Returns:
    - Zonotope representing the FFT component
    """
    N_data = len(Xk)
    
    ks = np.arange(N_data)
    thetas = 2 * np.pi / N_data * ks * h
    
    rot_matrices = np.array([[np.cos(theta), -np.sin(theta)] for theta in thetas])
    zk = [convert_interval_to_zonotope(x) for x in Xk]
    
    zk_rot = [zk[i].linear_map(np.array([[rot_matrices[i, 0]], [rot_matrices[i, 1]]])) for i in range(N_data)]
    zs = [overapproximate_zonotope(z) for z in zk_rot]
    
    z_out = minkowski_sum(zs[1], zs[0])
    for i in range(2, N_data):
        z_out = minkowski_sum(zs[i], z_out)
    
    return z_out


def inverse_intervalFFT_(Zh, k):
    """
    Compute a single component of the inverse FFT of a zonotope vector.
    
    Parameters:
    - Zh: List of zonotopes
    - k: Index
    
    Returns:
    - Zonotope representing the inverse FFT component
    """
    N_data = len(Zh)
    
    hs = np.arange(N_data)
    thetas = 2 * np.pi / N_data * hs * k
    
    rot_matrices = [np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ]) for theta in thetas]
    
    zh_rot = [zh.linear_map(rot_matrices[i]) for i, zh in enumerate(Zh)]
    zs = [overapproximate_zonotope(z) for z in zh_rot]
    
    z_out = minkowski_sum(zs[1], zs[0])
    for i in range(2, N_data):
        z_out = minkowski_sum(zs[i], z_out)
    
    return scale(1/N_data, z_out)


def intervalFFT(Xk):
    """
    Compute the FFT of an interval vector.
    
    Parameters:
    - Xk: List of interval objects
    
    Returns:
    - List of zonotopes representing the FFT
    """
    return [intervalFFT_(Xk, i) for i in range(len(Xk))]


def inverse_intervalFFT(Zh):
    """
    Compute the inverse FFT of a zonotope vector.
    
    Parameters:
    - Zh: List of zonotopes
    
    Returns:
    - List of zonotopes representing the inverse FFT
    """
    return [inverse_intervalFFT_(Zh, i) for i in range(len(Zh))]


def get_real(z):
    """
    Extract the real part interval from a zonotope.
    
    Parameters:
    - z: A zonotope
    
    Returns:
    - Interval representing the real part
    """
    z_highs = z.high()
    z_lows = z.low()
    return interval([z_lows[0], z_highs[0]])


def compute_inverse(kernel_fft, eps=1e-16):
    """
    Compute the inverse of a kernel in frequency domain.
    
    Parameters:
    - kernel_fft: FFT of the kernel
    - eps: Small number to avoid division by zero
    
    Returns:
    - Inverse of the kernel
    """
    return 1.0 / (kernel_fft + eps)


def set_PRE(neural_test):
    """
    Main function to compute set bounds using PRE method.
    
    Parameters:
    - neural_test: Neural network solution
    
    Returns:
    - List of intervals representing bounds
    """
    # Parameters
    m = 1
    k = 1
    dt = 0.1010101
    D_tt_kernel = np.array([1, -2, 1])
    D_identity = np.array([0, 1, 0])
    
    D_pos_kernel = m * D_tt_kernel + dt**2 * k * D_identity
    
    signal_padded = np.concatenate(([0], neural_test, [0]))
    
    N_signal = len(signal_padded)
    N_pad = N_signal - len(D_pos_kernel)
    kernel_pad = np.concatenate((D_pos_kernel, np.zeros(N_pad)))
    
    signal_fft = fft(signal_padded)
    kernel_fft = fft(kernel_pad)
    
    convolved = ifft(signal_fft * kernel_fft)
    inverse_kernel = compute_inverse(kernel_fft)
    
    convolved_noedges = convolved[4:-1]
    right_edges = convolved[1:4]
    left_edges = convolved[-1]
    
    # Create interval sets for different parts
    convolved_set_center = [interval([-abs(x.real), abs(x.real)]) for x in convolved_noedges]
    convolved_set_right = [interval([x.real, x.real]) for x in right_edges]
    convolved_set_left = [interval([left_edges.real, left_edges.real])]
    
    convolved_set = convolved_set_right + convolved_set_center + convolved_set_left
    
    # Perform interval FFT
    convolved_set_fft = intervalFFT(convolved_set)
    
    # Multiply with inverse kernel
    convolved_set_fft_kernel = [complex_prod(z, c) for z, c in zip(convolved_set_fft, inverse_kernel)]
    
    # Perform inverse interval FFT
    retrieved_signal = inverse_intervalFFT(convolved_set_fft_kernel)
    
    # Extract real parts
    return [get_real(z) for z in retrieved_signal]


def main():
    """Main function to run the PRE set propagation"""
    # Load data
    numerical_sol = np.load("ODE_outputs.npy")
    neural_sol = np.load("Nueral_outputs.npy")
    
    # Select a random ID
    ID = np.random.randint(0, 5)
    neural_test = neural_sol[ID, :, 0]
    numerical_test = numerical_sol[ID, :, 0]
    
    # Compute bounds
    signal_bounds = set_PRE(neural_test)
    signal_bounds_back = signal_bounds[1:-1]
    
    # Check if solutions are within bounds
    is_it_in_numerical = all(numerical_test[i] in signal_bounds_back[i] for i in range(len(numerical_test)))
    is_it_in_neural = all(neural_test[i] in signal_bounds_back[i] for i in range(len(neural_test)))
    
    print(f"Numerical is in: {is_it_in_numerical} and Neural is in: {is_it_in_neural}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    # Plot neural solution
    plt.plot(neural_test, label="neural")
    
    # Plot numerical solution
    plt.plot(numerical_test, label="numerical")
    
    # Plot bounds
    upper_bounds = [float(interval_obj.sup) for interval_obj in signal_bounds_back]
    lower_bounds = [float(interval_obj.inf) for interval_obj in signal_bounds_back]
    plt.fill_between(range(len(signal_bounds_back)), lower_bounds, upper_bounds, alpha=0.2)
    
    plt.legend()
    plt.title("ODE Solutions with PRE Set Bounds")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
