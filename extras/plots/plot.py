import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from numpy.linalg import inv

class LoopTransformationVisualizer:
    """
    A class for visualizing loop iteration spaces, dependencies, and transformations.
    """
    
    def __init__(self, loop_bounds, dependency_vectors=None):
        """
        Initialize the visualizer with loop bounds and optional dependency vectors.
        
        Parameters:
        -----------
        loop_bounds : tuple
            A tuple (N, M) specifying the bounds of the two nested loops.
        dependency_vectors : list of tuples, optional
            List of dependency vectors, each as a tuple (di, dj).
        """
        self.N, self.M = loop_bounds
        self.dependency_vectors = dependency_vectors or []
        self.original_points = np.array([(i, j) for i in range(self.N) for j in range(self.M)])
        self.transformed_points = self.original_points.copy()
        self.transformation_matrix = np.eye(2)  # Identity matrix by default
        self.transformation_offset = np.zeros(2)  # Zero offset by default
        
    def set_dependency_vectors(self, dependency_vectors):
        """
        Set dependency vectors.
        
        Parameters:
        -----------
        dependency_vectors : list of tuples
            List of dependency vectors, each as a tuple (di, dj).
        """
        self.dependency_vectors = dependency_vectors
        
    def apply_transformation(self, matrix=None, offset=None):
        """
        Apply an affine transformation to the iteration space.
        
        Parameters:
        -----------
        matrix : numpy.ndarray, optional
            2x2 transformation matrix. If None, uses identity matrix.
        offset : numpy.ndarray, optional
            Offset vector. If None, uses zero vector.
            
        The transformation applied is: (i,j) -> matrix @ (i,j) + offset
        """
        matrix = matrix if matrix is not None else np.eye(2)
        offset = offset if offset is not None else np.zeros(2)
        
        self.transformation_matrix = matrix
        self.transformation_offset = offset
        
        # Apply transformation to each point
        self.transformed_points = np.array([
            matrix @ point + offset for point in self.original_points
        ])
        
    def transform_point(self, point):
        """
        Transform a single point using the current transformation.
        
        Parameters:
        -----------
        point : tuple or numpy.ndarray
            The point (i, j) to transform.
            
        Returns:
        --------
        numpy.ndarray
            The transformed point.
        """
        return self.transformation_matrix @ point + self.transformation_offset
        
    def visualize(self, show_all_dependencies=False, arrow_stride=2, figsize=(16, 8)):
        """
        Visualize original and transformed iteration spaces with dependencies.
        
        Parameters:
        -----------
        show_all_dependencies : bool, optional
            If True, show dependencies for all points. If False, show them for a subset.
        arrow_stride : int, optional
            When show_all_dependencies is False, show arrows every arrow_stride steps.
        figsize : tuple, optional
            Figure size (width, height) in inches.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot original space
        ax1.scatter(self.original_points[:, 0], self.original_points[:, 1], color='blue', s=100)
        
        # Draw dependencies in original space
        for dep_vector in self.dependency_vectors:
            dep_vector = np.array(dep_vector)
            for idx, (i, j) in enumerate(self.original_points):
                if show_all_dependencies or (i % arrow_stride == 0 and j % arrow_stride == 0):
                    # Check if the target is within bounds
                    target_i, target_j = i + dep_vector[0], j + dep_vector[1]
                    if 0 <= target_i < self.N and 0 <= target_j < self.M:
                        arrow = FancyArrowPatch(
                            (i, j),
                            (target_i, target_j),
                            arrowstyle='->',
                            mutation_scale=15,
                            linewidth=1.5,
                            color='red'
                        )
                        ax1.add_patch(arrow)
        
        # Set up original space plot
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_xlabel('i (outer loop index)', fontsize=12)
        ax1.set_ylabel('j (inner loop index)', fontsize=12)
        ax1.set_title('Original Iteration Space with Dependencies', fontsize=14)
        ax1.set_xlim(-0.5, self.N - 0.5 + max(0, *[d[0] for d in self.dependency_vectors]))
        ax1.set_ylim(-0.5, self.M - 0.5 + max(0, *[d[1] for d in self.dependency_vectors]))
        ax1.set_xticks(range(self.N + 1))
        ax1.set_yticks(range(self.M + 1))
        
        # Plot transformed space
        ax2.scatter(self.transformed_points[:, 0], self.transformed_points[:, 1], color='green', s=100)
        
        # Draw dependencies in transformed space
        for dep_vector in self.dependency_vectors:
            dep_vector = np.array(dep_vector)
            for idx, (i, j) in enumerate(self.original_points):
                if show_all_dependencies or (i % arrow_stride == 0 and j % arrow_stride == 0):
                    # Check if the target is within bounds in the original space
                    target_i, target_j = i + dep_vector[0], j + dep_vector[1]
                    if 0 <= target_i < self.N and 0 <= target_j < self.M:
                        # Transform both source and target points
                        source_transformed = self.transform_point(np.array([i, j]))
                        target_transformed = self.transform_point(np.array([target_i, target_j]))
                        
                        arrow = FancyArrowPatch(
                            tuple(source_transformed),
                            tuple(target_transformed),
                            arrowstyle='->',
                            mutation_scale=15,
                            linewidth=1.5,
                            color='red'
                        )
                        ax2.add_patch(arrow)
        
        # Set up transformed space plot
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_xlabel('Transformed i-axis', fontsize=12)
        ax2.set_ylabel('Transformed j-axis', fontsize=12)
        
        # Create a string representation of the transformation
        matrix_str = np.array2string(self.transformation_matrix, precision=2, separator=', ')
        offset_str = np.array2string(self.transformation_offset, precision=2, separator=', ')
        transform_formula = f"T(i,j) = {matrix_str} @ (i,j) + {offset_str}"
        
        ax2.set_title(f'Transformed Iteration Space\n{transform_formula}', fontsize=14)
        
        # Set limits for the transformed space with padding
        min_x = np.min(self.transformed_points[:, 0]) - 0.5
        max_x = np.max(self.transformed_points[:, 0]) + 0.5
        min_y = np.min(self.transformed_points[:, 1]) - 0.5
        max_y = np.max(self.transformed_points[:, 1]) + 0.5
        
        ax2.set_xlim(min_x, max_x)
        ax2.set_ylim(min_y, max_y)
        
        # Try to set sensible tick marks for transformed space
        x_range = max_x - min_x
        y_range = max_y - min_y
        
        if x_range <= 20:  # Use integer ticks if range is reasonable
            ax2.set_xticks(np.arange(np.floor(min_x), np.ceil(max_x) + 1))
        
        if y_range <= 20:
            ax2.set_yticks(np.arange(np.floor(min_y), np.ceil(max_y) + 1))
        
        # Add legend
        red_line = plt.Line2D([0], [0], color='red', lw=1.5, marker='>', markersize=8)
        
        if len(self.dependency_vectors) == 1:
            dep_str = f"Dependency: (i,j) → (i+{self.dependency_vectors[0][0]},j+{self.dependency_vectors[0][1]})"
            ax1.legend([red_line], [dep_str], loc='upper right')
        else:
            ax1.legend([red_line], ['Dependencies'], loc='upper right')
            
        ax2.legend([red_line], ['Transformed Dependencies'], loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
    def skew_transformation(self, skew_factor_i=0, skew_factor_j=0):
        """
        Apply a skew transformation.
        
        Parameters:
        -----------
        skew_factor_i : float
            Skew factor for i coordinate (horizontal shearing).
        skew_factor_j : float
            Skew factor for j coordinate (vertical shearing).
        """
        matrix = np.array([
            [1, skew_factor_i],
            [skew_factor_j, 1]
        ])
        self.apply_transformation(matrix)
        
    def rotate_transformation(self, angle_degrees):
        """
        Apply a rotation transformation.
        
        Parameters:
        -----------
        angle_degrees : float
            Rotation angle in degrees.
        """
        angle_radians = np.deg2rad(angle_degrees)
        cos_theta = np.cos(angle_radians)
        sin_theta = np.sin(angle_radians)
        
        matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        self.apply_transformation(matrix)
        
    def scale_transformation(self, scale_i=1, scale_j=1):
        """
        Apply a scaling transformation.
        
        Parameters:
        -----------
        scale_i : float
            Scale factor for i coordinate.
        scale_j : float
            Scale factor for j coordinate.
        """
        matrix = np.array([
            [scale_i, 0],
            [0, scale_j]
        ])
        self.apply_transformation(matrix)
        
    def custom_transformation(self, a, b, c, d, offset_i=0, offset_j=0):
        """
        Apply a custom 2x2 linear transformation: (i,j) -> (a*i + b*j, c*i + d*j) + (offset_i, offset_j)
        
        Parameters:
        -----------
        a, b, c, d : float
            Elements of the transformation matrix.
        offset_i, offset_j : float
            Elements of the offset vector.
        """
        matrix = np.array([[a, b], [c, d]])
        offset = np.array([offset_i, offset_j])
        self.apply_transformation(matrix, offset)
        
    def compose_transformation(self, new_matrix, new_offset=None):
        """
        Compose the current transformation with a new transformation.
        
        Parameters:
        -----------
        new_matrix : numpy.ndarray
            The new 2x2 transformation matrix to compose with the current one.
        new_offset : numpy.ndarray, optional
            The new offset vector. If None, uses zero vector.
        
        Note: The new transformation is applied after the current one.
        """
        new_offset = new_offset if new_offset is not None else np.zeros(2)
        
        # Compute the composed transformation: new_matrix @ (old_matrix @ x + old_offset) + new_offset
        # This simplifies to (new_matrix @ old_matrix) @ x + (new_matrix @ old_offset + new_offset)
        composed_matrix = new_matrix @ self.transformation_matrix
        composed_offset = new_matrix @ self.transformation_offset + new_offset
        
        self.apply_transformation(composed_matrix, composed_offset)
        
    def invert_transformation(self):
        """
        Invert the current transformation if possible.
        
        Returns:
        --------
        bool
            True if inversion was successful, False otherwise.
        """
        try:
            inverted_matrix = inv(self.transformation_matrix)
            inverted_offset = -inverted_matrix @ self.transformation_offset
            self.apply_transformation(inverted_matrix, inverted_offset)
            return True
        except np.linalg.LinAlgError:
            print("Error: Transformation is not invertible.")
            return False

# Create a visualizer
visualizer = LoopTransformationVisualizer((5, 7), [(1, 0),(1,1),(1,-1)])

# Apply skew: (i,j) → (i-j, j)
visualizer.custom_transformation(0, 1, 1, 0)  # Matrix = [[1, -1], [0, 1]]
visualizer.custom_transformation(1, 1, 1, 0)  # Matrix = [[1, -1], [0, 1]]

# Visualize
visualizer.visualize(show_all_dependencies=True)
