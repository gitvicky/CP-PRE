import numpy as np
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
        
        # For efficiency, if we have too many generators, we'll take a subset
        if n_generators > 10:  # Arbitrary threshold to avoid combinatorial explosion
            # This is a simplification - a real implementation would handle this differently
            potential_vertices = self._sample_vertices(1000)
        else:
            coeffs = np.array(np.meshgrid(*[[-1, 1] for _ in range(n_generators)]))
            coeffs = coeffs.T.reshape(-1, n_generators)
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
    
    def _sample_vertices(self, n_samples):
        """
        Sample vertices from the zonotope for cases with many generators.
        
        Parameters:
        - n_samples: Number of vertices to sample
        
        Returns:
        - Sample of vertices
        """
        n_generators = self.generators.shape[1]
        coeffs = np.random.uniform(-1, 1, (n_samples, n_generators))
        coeffs = np.sign(coeffs)  # Convert to -1 or 1
        return self.center + np.dot(coeffs, self.generators.T)
    
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
            A_eq = self.generators.T
            b_eq = shifted
            bounds = [(-1, 1) for _ in range(self.generators.shape[1])]
            
            # Handle underconstrained system
            if A_eq.shape[0] < A_eq.shape[1]:
                result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
                return result.success and np.max(np.abs(result.x)) <= 1
            else:
                # For overconstrained systems, check if there's a feasible solution
                residual = np.linalg.lstsq(A_eq, b_eq, rcond=None)[1]
                if residual.size == 0 or residual[0] < 1e-10:
                    # The point is approximately on a linear combination of generators
                    return True
                return False
        except:
            # Fallback to a simple approximation
            box_high = self.high()
            box_low = self.low()
            return np.all(point >= box_low) and np.all(point <= box_high)
            
    def reduce_generators(self, max_generators=20):
        """
        Reduce the number of generators by combining similar ones.
        This is a simplified approach for demonstration.
        
        Parameters:
        - max_generators: Maximum number of generators to keep
        
        Returns:
        - A new zonotope with reduced generators
        """
        if self.generators.shape[1] <= max_generators:
            return self
        
        # Simple strategy: combine generators with similar directions
        from sklearn.cluster import KMeans
        
        # Normalize generators
        norms = np.linalg.norm(self.generators, axis=0)
        normalized_gens = self.generators / (norms + 1e-10)
        
        # Cluster generators
        kmeans = KMeans(n_clusters=max_generators).fit(normalized_gens.T)
        
        # Create new generators based on cluster centers
        new_generators = np.zeros((self.dim, max_generators))
        for i in range(max_generators):
            mask = kmeans.labels_ == i
            gen_sum = np.sum(self.generators[:, mask], axis=1)
            new_generators[:, i] = gen_sum
        
        return Zonotope(self.center, new_generators)
