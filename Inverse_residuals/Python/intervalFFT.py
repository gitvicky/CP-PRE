import numpy as np
from zonopy import interval, zonotope


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
    
    Z_rot = linear_map(rot_matrix, Z)
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
    inf_val = float(intv.inf)
    sup_val = float(intv.sup)
    
    # Create center and generator
    center = np.array([(inf_val + sup_val) / 2, 0])
    
    # Create generator matrix
    radius = (sup_val - inf_val) / 2
    generators = np.array([[radius, 0], [0, 0]]).T

    return zonotope(np.vstack([center, generators]))


def minkowski_sum(Z1, Z2):
    """
    Compute the Minkowski sum of two zonotopes.
    
    Parameters:
    - Z1, Z2: Two zonotopes
    
    Returns:
    - Their Minkowski sum
    """
    return Z1 + Z2


def linear_map(M, Z):

    c = np.matmul(M, Z.center)
    G = np.matmul(M, Z.generators.T)
    return zonotope(np.vstack([c, G.T]))


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
        Zk_rot.append(linear_map(matrix, Zk[i]))
    
    # Compute Minkowski sum
    Z_out = minkowski_sum(Zk_rot[1], Zk_rot[0])
    for i in range(2, N_data):
        Z_out = minkowski_sum(Zk_rot[i], Z_out)
    
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
    Zh_rot = [linear_map(rot_matrices[i], Zh[i]) for i in range(N_data)]
    
    # Compute Minkowski sum
    Z_out = minkowski_sum(Zh_rot[1], Zh_rot[0])
    for i in range(2, N_data):
        Z_out = minkowski_sum(Zh_rot[i], Z_out)
    
    return 1/N_data * Z_out


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
    Z_interval = Z.to_interval()
    
    return Z_interval[0]


def box(Z):
    """
    Extract the bounding box of a zonotope as intervals.
    
    Parameters:
    - Z: A zonotope
    
    Returns:
    - Two intervals: real and imaginary parts
    """

    return Z.to_interval()


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