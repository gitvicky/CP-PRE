import numpy as np
from interval import interval, imath
from zonotope import Zonotope


def complex_prod(Z, C):
    """
    Multiply a zonotopic complex number (represented as a 2D zonotope)
    by a precise complex number C.
    
    Parameters:
    - Z: A zonotope representing a complex number (2D)
    - C: A complex number
    
    Returns:
    - A zonotope representing the product
    """
    scaling_fac = abs(C)
    angle = np.arctan2(C.imag, C.real)
    
    rot_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    Z_rot = Z.linear_map(rot_matrix)
    return scaling_fac * Z_rot


def convert_interval_to_zonotope(intv):
    """
    Convert a Python interval to a zonotope in 2D space (real and imaginary parts).
    
    Parameters:
    - intv: An interval object
    
    Returns:
    - A zonotope representing the interval on the real axis
    """
    # Extract lower and upper bounds
    inf_val = float(intv[0][0])
    sup_val = float(intv[0][1])
    
    # Create center and generator
    center = np.array([(inf_val + sup_val) / 2, 0])
    
    # Create generator matrix
    radius = (sup_val - inf_val) / 2
    generators = np.array([[radius, 0], [0, 0]]).T
    
    return Zonotope(center, generators)


def overapproximate(Z, target_type=None):
    """
    Overapproximate a zonotope, potentially with fewer generators.
    In this implementation, we just return the original zonotope or call 
    the reduce_generators method if the zonotope has too many generators.
    
    Parameters:
    - Z: A zonotope
    - target_type: Placeholder for compatibility with Julia code
    
    Returns:
    - An overapproximated zonotope
    """
    if Z.generators.shape[1] > 50:  # Arbitrary threshold
        return Z.reduce_generators(30)  # Reduce to 30 generators
    return Z


def minkowski_sum(Z1, Z2):
    """
    Compute the Minkowski sum of two zonotopes.
    
    Parameters:
    - Z1, Z2: Two zonotopes
    
    Returns:
    - Their Minkowski sum
    """
    return Z1 + Z2


def scale(factor, Z):
    """
    Scale a zonotope by a factor.
    
    Parameters:
    - factor: A scalar
    - Z: A zonotope
    
    Returns:
    - The scaled zonotope
    """
    return factor * Z


def intervalFFT_(Xk, h):
    """
    Compute a single component of the FFT of a vector of intervals.
    
    Parameters:
    - Xk: A list of intervals
    - h: The index
    
    Returns:
    - A zonotope representing the FFT component
    """
    N_data = len(Xk)
    
    ks = np.arange(N_data)
    thetas = 2 * np.pi / N_data * ks * h
    
    # Create rotation matrices
    rot_matrices = np.array([
        np.array([[np.cos(theta)], [-np.sin(theta)]]) for theta in thetas
    ])
    
    # Convert intervals to zonotopes
    Zk = [convert_interval_to_zonotope(x) for x in Xk]
    
    # Apply rotations
    Zk_rot = []
    for i in range(N_data):
        matrix = np.array([
            [rot_matrices[i][0][0], 0],
            [rot_matrices[i][1][0], 0]
        ])
        Zk_rot.append(Zk[i].linear_map(matrix))
    
    # Overapproximate
    Zs = [overapproximate(z) for z in Zk_rot]
    
    # Compute Minkowski sum
    Z_out = minkowski_sum(Zs[1], Zs[0])
    for i in range(2, N_data):
        Z_out = minkowski_sum(Zs[i], Z_out)
        
        # Periodically reduce generators to manage complexity
        if i % 10 == 0:
            Z_out = overapproximate(Z_out)
    
    return Z_out


def inverse_intervalFFT_(Zh, k):
    """
    Compute a single component of the inverse FFT of a vector of zonotopes.
    
    Parameters:
    - Zh: A list of zonotopes
    - k: The index
    
    Returns:
    - A zonotope representing the inverse FFT component
    """
    N_data = len(Zh)
    
    hs = np.arange(N_data)
    thetas = 2 * np.pi / N_data * hs * k
    
    # Create rotation matrices
    rot_matrices = [
        np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ]) for theta in thetas
    ]
    
    # Apply rotations
    Zh_rot = [Zh[i].linear_map(rot_matrices[i]) for i in range(N_data)]
    
    # Overapproximate
    Zs = [overapproximate(z) for z in Zh_rot]
    
    # Compute Minkowski sum
    Z_out = minkowski_sum(Zs[1], Zs[0])
    for i in range(2, N_data):
        Z_out = minkowski_sum(Zs[i], Z_out)
        
        # Periodically reduce generators to manage complexity
        if i % 10 == 0:
            Z_out = overapproximate(Z_out)
    
    return scale(1/N_data, Z_out)


def intervalFFT(Xk):
    """
    Compute the FFT of a vector of intervals.
    
    Parameters:
    - Xk: A list of intervals
    
    Returns:
    - A list of zonotopes representing the FFT
    """
    return [intervalFFT_(Xk, i) for i in range(len(Xk))]


def inverse_intervalFFT(Zh):
    """
    Compute the inverse FFT of a vector of zonotopes.
    
    Parameters:
    - Zh: A list of zonotopes
    
    Returns:
    - A list of zonotopes representing the inverse FFT
    """
    return [inverse_intervalFFT_(Zh, i) for i in range(len(Zh))]


def Real(Z):
    """
    Extract the real part interval from a zonotope.
    
    Parameters:
    - Z: A zonotope
    
    Returns:
    - An interval representing the real part
    """
    Z_highs = Z.high()
    Z_lows = Z.low()
    
    return interval([Z_lows[0], Z_highs[0]])


def box(Z):
    """
    Extract the bounding box of a zonotope as intervals.
    
    Parameters:
    - Z: A zonotope
    
    Returns:
    - Two intervals: real and imaginary parts
    """
    Z_highs = Z.high()
    Z_lows = Z.low()
    
    Z_real = interval([Z_lows[0], Z_highs[0]])
    Z_imag = interval([Z_lows[1], Z_highs[1]])
    
    return Z_real, Z_imag


def amplitude(Z):
    """
    Compute the amplitude (norm) interval of a zonotope.
    
    Parameters:
    - Z: A zonotope representing a complex number
    
    Returns:
    - An interval for the amplitude
    """
    vertices_list = Z.vertices()
    amplitudes = np.linalg.norm(vertices_list, axis=1)
    
    # Check if zero is contained
    if Z.contains([0, 0]):
        return interval([0, np.max(amplitudes)])
    
    return interval([np.min(amplitudes), np.max(amplitudes)])