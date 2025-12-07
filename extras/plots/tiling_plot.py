import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Polygon
from matplotlib.colors import to_rgba
from numpy.linalg import inv
import matplotlib.colors as mcolors
from matplotlib.path import Path


class LoopTransformationVisualizer:
    """
    A class for visualizing loop iteration spaces, dependencies, transformations, and tiling.
    """

    def __init__(self, loop_bounds, dependency_vectors=None, show_all=False):
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
        self.original_points = np.array(
            [(i, j) for i in range(self.N) for j in range(self.M)]
        )
        self.transformed_points = self.original_points.copy()
        self.transformation_matrix = np.eye(2)  # Identity matrix by default
        self.transformation_offset = np.zeros(2)  # Zero offset by default
        self.show_all = (show_all,)

        # Tiling parameters
        self.is_tiled = False
        self.tile_sizes = None
        self.tiles = None  # Will store (tile_i, tile_j) -> [(point_i, point_j), ...]
        self.tile_mapping = None  # Will store (point_i, point_j) -> (tile_i, tile_j)

        # Transformed dependencies
        self.transformed_deps = self.dependency_vectors.copy()

        # Tile dependencies (if tiling is applied)
        self.tile_deps = []

        # Transformation history for visualization
        self.transformation_history = []

    def set_dependency_vectors(self, dependency_vectors):
        """
        Set dependency vectors.

        Parameters:
        -----------
        dependency_vectors : list of tuples
            List of dependency vectors, each as a tuple (di, dj).
        """
        self.dependency_vectors = dependency_vectors
        self.transformed_deps = dependency_vectors.copy()
        self.tile_deps = []

    def apply_transformation(
        self, matrix=None, offset=None, name="Custom Transformation"
    ):
        """
        Apply an affine transformation to the iteration space.

        Parameters:
        -----------
        matrix : numpy.ndarray, optional
            2x2 transformation matrix. If None, uses identity matrix.
        offset : numpy.ndarray, optional
            Offset vector. If None, uses zero vector.
        name : str, optional
            Name of the transformation for visualization purposes.

        The transformation applied is: (i,j) -> matrix @ (i,j) + offset
        """
        matrix = matrix if matrix is not None else np.eye(2)
        offset = offset if offset is not None else np.zeros(2)

        # Store the current state in history
        self.transformation_history.append(
            {
                "name": name,
                "points": self.transformed_points.copy(),
                "matrix": self.transformation_matrix.copy(),
                "offset": self.transformation_offset.copy(),
                "is_tiled": self.is_tiled,
                "tile_sizes": self.tile_sizes,
                "transformed_deps": self.transformed_deps.copy(),
                "tile_deps": self.tile_deps.copy() if self.tile_deps else [],
            }
        )

        # Update transformation matrix and offset
        new_matrix = matrix @ self.transformation_matrix
        new_offset = matrix @ self.transformation_offset + offset

        self.transformation_matrix = new_matrix
        self.transformation_offset = new_offset

        # Transform all points
        self.transformed_points = np.array(
            [new_matrix @ point + offset for point in self.original_points]
        )

        # Transform dependencies
        self.transformed_deps = [
            tuple(new_matrix @ np.array(dep)) for dep in self.dependency_vectors
        ]

        # If tiled, update tile information
        if self.is_tiled:
            self._update_tiles_after_transformation(matrix, offset)

        # Visualize the transformation
        self.visualize_transformation(name)

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
        return self.transformation_matrix @ np.array(point) + self.transformation_offset

    def apply_tiling(
        self, tile_sizes, name="Tiling Transformation", show_point_deps=False
    ):
        """
        Apply tiling to the iteration space.

        Parameters:
        -----------
        tile_sizes : tuple
            A tuple (ti, tj) specifying the size of tiles in i and j dimensions.
        name : str, optional
            Name of the transformation for visualization purposes.
        show_point_deps : bool, optional
            If False, hide point-level dependencies after tiling and show only tile dependencies.
        """
        # Store current state in history
        self.transformation_history.append(
            {
                "name": name,
                "points": self.transformed_points.copy(),
                "matrix": self.transformation_matrix.copy(),
                "offset": self.transformation_offset.copy(),
                "is_tiled": self.is_tiled,
                "tile_sizes": self.tile_sizes,
                "transformed_deps": self.transformed_deps.copy(),
                "tile_deps": self.tile_deps.copy() if self.tile_deps else [],
            }
        )

        self.is_tiled = True
        self.tile_sizes = tile_sizes
        self.show_point_deps = show_point_deps  # Store this preference

        # Create tile groupings
        ti, tj = tile_sizes
        self.tiles = {}
        self.tile_mapping = {}

        # Group points into tiles based on transformed coordinates
        for idx, point in enumerate(self.transformed_points):
            i, j = point
            tile_i, tile_j = int(i // ti), int(j // tj)

            # Store point in tile
            tile_key = (tile_i, tile_j)
            if tile_key not in self.tiles:
                self.tiles[tile_key] = []
            self.tiles[tile_key].append(tuple(point))

            # Map point to its tile
            self.tile_mapping[tuple(point)] = tile_key

        # Convert point dependencies to tile dependencies
        self.tile_deps = self._convert_to_tile_dependencies(self.transformed_deps)

        # Visualize the tiling
        self.visualize_transformation(name)

    def _convert_to_tile_dependencies(self, point_deps):
        """
        Convert point dependencies to tile dependencies after tiling.

        Parameters:
        -----------
        point_deps : list of tuples
            List of point dependencies.

        Returns:
        --------
        list of tuples
            List of tile dependencies.
        """
        if not self.is_tiled:
            return []

        ti, tj = self.tile_sizes
        tile_deps = set()

        for dep in point_deps:
            di, dj = dep

            # Calculate possible tile dependencies
            i_min = int(np.floor(di / ti))
            i_max = int(np.ceil(di / ti)) if di % ti != 0 else i_min

            j_min = int(np.floor(dj / tj))
            j_max = int(np.ceil(dj / tj)) if dj % tj != 0 else j_min

            # Add all possible tile dependencies
            for i_offset in range(i_min, i_max + 1):
                for j_offset in range(j_min, j_max + 1):
                    if i_offset != 0 or j_offset != 0:  # Skip (0,0)
                        tile_deps.add((i_offset, j_offset))

        return list(tile_deps)

    def _update_tiles_after_transformation(self, matrix, offset):
        """
        Update tile information after a transformation is applied.

        Parameters:
        -----------
        matrix : numpy.ndarray
            Transformation matrix.
        offset : numpy.ndarray
            Offset vector.
        """
        if not self.is_tiled:
            return

        # Transform tile keys (conceptually, tile origins) directly
        new_tiles = {}
        new_tile_mapping = {}

        for tile_key, points in self.tiles.items():
            # Calculate new tile key after transformation
            tile_i, tile_j = tile_key
            new_tile_key = tuple(matrix @ np.array([tile_i, tile_j]))

            # Add all points from the old tile to the new tile
            new_tiles[new_tile_key] = []
            for point in points:
                new_point = tuple(matrix @ np.array(point) + offset)
                new_tiles[new_tile_key].append(new_point)
                new_tile_mapping[new_point] = new_tile_key

        self.tiles = new_tiles
        self.tile_mapping = new_tile_mapping

        # Update tile dependencies
        self.tile_deps = [tuple(matrix @ np.array(dep)) for dep in self.tile_deps]

    def visualize_transformation(
        self, name="Transformation", figsize=(16, 8), zoom_factor=1.0
    ):
        """
        Visualize the current state of the iteration space after a transformation.

        Parameters:
        -----------
        name : str, optional
            Name of the transformation for the plot title.
        figsize : tuple, optional
            Figure size (width, height) in inches.
        zoom_factor : float, optional
            Zoom factor to control the view (>1 zooms in, <1 zooms out).
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot original space on the left
        ax1.scatter(
            self.original_points[:, 0], self.original_points[:, 1], color="blue", s=50
        )

        # Draw dependencies in original space
        for dep_vector in self.dependency_vectors:
            for i, j in self.original_points:
                if self.show_all or  i % 2 == 0 and j % 2 == 0:  # Show only some arrows to avoid clutter
                    target_i, target_j = i + dep_vector[0], j + dep_vector[1]
                    if 0 <= target_i < self.N and 0 <= target_j < self.M:
                        arrow = FancyArrowPatch(
                            (i, j),
                            (target_i, target_j),
                            arrowstyle="->",
                            mutation_scale=15,
                            linewidth=1.5,
                            color="red",
                        )
                        ax1.add_patch(arrow)

        ax1.set_title("Original Iteration Space with Dependencies", fontsize=14)
        ax1.set_xlabel("i (outer loop index)", fontsize=12)
        ax1.set_ylabel("j (inner loop index)", fontsize=12)
        ax1.grid(True, linestyle="--", alpha=0.7)

        # Set limits with padding
        ax1.set_xlim(-0.5, self.N - 0.5)
        ax1.set_ylim(-0.5, self.M - 0.5)

        # Plot transformed space on the right
        ax2.scatter(
            self.transformed_points[:, 0],
            self.transformed_points[:, 1],
            color="green",
            s=50,
        )

        # Draw dependencies in transformed space - ONLY IF not tiled OR show_point_deps is True
        show_point_dependencies = not self.is_tiled or getattr(
            self, "show_point_deps", True
        )

        if show_point_dependencies:
            for dep_idx, dep_vector in enumerate(self.transformed_deps):
                original_dep = self.dependency_vectors[dep_idx]
                for idx, (i, j) in enumerate(self.transformed_points):
                    if self.show_all or  idx % 4 == 0:  # Show only some arrows to avoid clutter
                        # Calculate target point in transformed space
                        target = np.array([i, j]) + np.array(dep_vector)

                        # Create arrow
                        arrow = FancyArrowPatch(
                            (i, j),
                            tuple(target),
                            arrowstyle="->",
                            mutation_scale=15,
                            linewidth=1.5,
                            color="red",
                        )
                        ax2.add_patch(arrow)

        # Draw tiles if tiling is applied
        if self.is_tiled:
            # Use different colors for different tiles
            colors = list(mcolors.TABLEAU_COLORS.values())
            n_colors = len(colors)

            for idx, (tile_key, points) in enumerate(self.tiles.items()):
                tile_i, tile_j = tile_key
                color = colors[idx % n_colors]

                # Convert tile points to numpy array for polygon
                if len(points) > 2:
                    # Use convex hull to create tile polygon
                    points_array = np.array(points)

                    # For simplicity, just use the min/max coordinates to create a bounding box
                    min_i, min_j = np.min(points_array, axis=0) - 0.5
                    max_i, max_j = np.max(points_array, axis=0) + 0.5

                    corners = [
                        [min_i, min_j],
                        [min_i, max_j],
                        [max_i, max_j],
                        [max_i, min_j],
                    ]

                    polygon = Polygon(
                        corners,
                        closed=True,
                        alpha=0.3,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=2,
                    )
                    ax2.add_patch(polygon)

                    # Add tile index label
                    center_i = (min_i + max_i) / 2
                    center_j = (min_j + max_j) / 2

                    ax2.text(
                        center_i,
                        center_j,
                        f"({tile_i},{tile_j})",
                        ha="center",
                        va="center",
                        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),
                        fontsize=10,
                        fontweight="bold",
                    )

            # Draw tile dependencies if present
            if self.tile_deps:
                # Find all tile centers
                tile_centers = {}
                for tile_key, points in self.tiles.items():
                    if points:
                        points_array = np.array(points)
                        center = np.mean(points_array, axis=0)
                        tile_centers[tile_key] = center

                # Draw dependency arrows between tile centers
                for base_tile in tile_centers:
                    base_center = tile_centers[base_tile]

                    # Draw arrows to dependent tiles
                    for tile_dep in self.tile_deps:
                        # Calculate target tile coordinates
                        target_tile = tuple(np.array(base_tile) + np.array(tile_dep))

                        if target_tile in tile_centers:
                            target_center = tile_centers[target_tile]

                            # Draw arrow
                            arrow = FancyArrowPatch(
                                tuple(base_center),
                                tuple(target_center),
                                arrowstyle="-|>",
                                mutation_scale=20,
                                linewidth=2.5,
                                color="purple",
                            )
                            ax2.add_patch(arrow)

                            # Add label
                            midpoint = (base_center + target_center) / 2
                            # ax2.text(
                            #     midpoint[0],
                            #     midpoint[1],
                            #     f"({tile_dep[0]},{tile_dep[1]})",
                            #     ha="center",
                            #     va="center",
                            #     bbox=dict(
                            #         facecolor="lightyellow", alpha=0.9, boxstyle="round"
                            #     ),
                            #     fontsize=10,
                            # )

        # Set title and labels for transformed space
        matrix_str = np.array2string(
            self.transformation_matrix, precision=2, separator=", "
        )
        title = f"{name}\nTransformation Matrix:\n{matrix_str}"
        ax2.set_title(title, fontsize=14)
        ax2.set_xlabel("Transformed i-axis", fontsize=12)
        ax2.set_ylabel("Transformed j-axis", fontsize=12)
        ax2.grid(True, linestyle="--", alpha=0.7)

        # Set limits with padding for transformed space
        min_x = np.min(self.transformed_points[:, 0]) - 0.5
        max_x = np.max(self.transformed_points[:, 0]) + 0.5
        min_y = np.min(self.transformed_points[:, 1]) - 0.5
        max_y = np.max(self.transformed_points[:, 1]) + 0.5

        # Extend limits to include arrows and dependencies
        if hasattr(self, "custom_limits") and self.custom_limits:
            if self.custom_limits["for_transformed"]:
                ax2.set_xlim(self.custom_limits["x_min"], self.custom_limits["x_max"])
                ax2.set_ylim(self.custom_limits["y_min"], self.custom_limits["y_max"])
            else:
                ax1.set_xlim(self.custom_limits["x_min"], self.custom_limits["x_max"])
                ax1.set_ylim(self.custom_limits["y_min"], self.custom_limits["y_max"])
        else:
            # Apply zoom factor to limits
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            half_width = (max_x - min_x) / 2
            half_height = (max_y - min_y) / 2

            ax2.set_xlim(
                center_x - half_width / zoom_factor, center_x + half_width / zoom_factor
            )
            ax2.set_ylim(
                center_y - half_height / zoom_factor,
                center_y + half_height / zoom_factor,
            )

        # Add legend
        red_line = plt.Line2D([0], [0], color="red", lw=1.5, marker=">", markersize=8)
        blue_dot = plt.Line2D(
            [0], [0], marker="o", color="blue", linestyle="", markersize=8
        )
        green_dot = plt.Line2D(
            [0], [0], marker="o", color="green", linestyle="", markersize=8
        )

        legend_items = [blue_dot, green_dot]
        legend_labels = ["Original Points", "Transformed Points"]

        if show_point_dependencies:
            legend_items.insert(0, red_line)
            legend_labels.insert(0, "Point Dependencies")

        if self.is_tiled:
            tile_patch = plt.Rectangle((0, 0), 1, 1, fc="blue", alpha=0.3)
            legend_items.append(tile_patch)
            legend_labels.append(f"Tiles ({self.tile_sizes[0]}x{self.tile_sizes[1]})")

            if self.tile_deps:
                tile_arrow = plt.Line2D(
                    [0], [0], color="purple", lw=2, marker=">", markersize=10
                )
                legend_items.append(tile_arrow)
                legend_labels.append("Tile Dependencies")

        ax2.legend(
            legend_items, legend_labels, loc="upper left", bbox_to_anchor=(1.05, 1)
        )

        # Add a textbox with transformation details
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

        # Create transformation description
        transform_desc = "Transformation Details:\n\n"
        transform_desc += f"Matrix = {matrix_str}\n"

        if self.transformation_offset.any():
            offset_str = np.array2string(
                self.transformation_offset, precision=2, separator=", "
            )
            transform_desc += f"Offset = {offset_str}\n"

        # Add dependency transformation details
        if show_point_dependencies:
            transform_desc += "\nPoint Dependency Transformations:\n"
            for i, (orig_dep, trans_dep) in enumerate(
                zip(self.dependency_vectors, self.transformed_deps)
            ):
                transform_desc += f"  • ({orig_dep[0]},{orig_dep[1]}) → ({trans_dep[0]:.1f},{trans_dep[1]:.1f})\n"

        # Add tile dependency details if tiled
        if self.is_tiled and self.tile_deps:
            transform_desc += f"\nTile Dependencies ({self.tile_sizes[0]}x{self.tile_sizes[1]} tiles):\n"
            for tile_dep in sorted(self.tile_deps):
                transform_desc += f"  • Tile({tile_dep[0]:.1f},{tile_dep[1]:.1f})\n"

        # Add the text box
        fig.text(
            0.02,
            0.5,
            transform_desc,
            transform=fig.transFigure,
            fontsize=10,
            verticalalignment="center",
            bbox=props,
        )

        plt.tight_layout()
        plt.subplots_adjust(
            right=0.85, left=0.3
        )  # Make room for the legend and textbox
        plt.show()

    def skew_transformation(
        self, skew_factor_i=0, skew_factor_j=0, name="Skew Transformation"
    ):
        """
        Apply a skew transformation.

        Parameters:
        -----------
        skew_factor_i : float
            Skew factor for i coordinate (horizontal shearing).
        skew_factor_j : float
            Skew factor for j coordinate (vertical shearing).
        name : str, optional
            Name of the transformation for visualization purposes.
        """
        matrix = np.array([[1, skew_factor_i], [skew_factor_j, 1]])
        self.apply_transformation(matrix, name=name)

    def custom_transformation(
        self, a, b, c, d, offset_i=0, offset_j=0, name="Custom Transformation"
    ):
        """
        Apply a custom 2x2 linear transformation: (i,j) -> (a*i + b*j, c*i + d*j) + (offset_i, offset_j)

        Parameters:
        -----------
        a, b, c, d : float
            Elements of the transformation matrix.
        offset_i, offset_j : float
            Elements of the offset vector.
        name : str, optional
            Name of the transformation for visualization purposes.
        """
        matrix = np.array([[a, b], [c, d]])
        offset = np.array([offset_i, offset_j])
        self.apply_transformation(matrix, offset, name=name)

    def visualize_history(self):
        """
        Visualize the transformation history as a sequence of plots.
        """
        if not self.transformation_history:
            print("No transformation history available.")
            return

        for idx, state in enumerate(self.transformation_history):
            print(f"Step {idx + 1}: {state['name']}")

            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot points
            ax.scatter(
                state["points"][:, 0], state["points"][:, 1], color="green", s=50
            )

            # Draw dependencies
            trans_deps = state["transformed_deps"]
            for dep_vector in trans_deps:
                for i, j in state["points"][
                    ::4
                ]:  # Plot every 4th point to avoid clutter
                    arrow = FancyArrowPatch(
                        (i, j),
                        (i + dep_vector[0], j + dep_vector[1]),
                        arrowstyle="->",
                        mutation_scale=15,
                        linewidth=1.5,
                        color="red",
                    )
                    ax.add_patch(arrow)

            # Draw tiles if tiled
            if state["is_tiled"]:
                # This is a simplified approach since we don't have the full tile information
                # from history. For a complete implementation, you would need to store more data.
                ti, tj = state["tile_sizes"]
                min_i = min(state["points"][:, 0])
                min_j = min(state["points"][:, 1])

                # Draw grid lines for tiles
                for i in range(int(min_i), int(max(state["points"][:, 0]) + ti), ti):
                    ax.axvline(i, color="blue", linestyle="--", alpha=0.3)
                for j in range(int(min_j), int(max(state["points"][:, 1]) + tj), tj):
                    ax.axhline(j, color="blue", linestyle="--", alpha=0.3)

                # Draw tile dependencies if available
                if state["tile_deps"]:
                    # This is a simplified visualization
                    for tile_dep in state["tile_deps"]:
                        ax.text(
                            min_i + 1,
                            min_j + 1,
                            f"Tile Dep: {tile_dep}",
                            ha="left",
                            va="bottom",
                            bbox=dict(
                                facecolor="lightyellow", alpha=0.9, boxstyle="round"
                            ),
                            fontsize=10,
                        )

            # Set title and labels
            matrix_str = np.array2string(state["matrix"], precision=2, separator=", ")
            ax.set_title(f"{state['name']}\nMatrix: {matrix_str}", fontsize=14)
            ax.set_xlabel("i-axis", fontsize=12)
            ax.set_ylabel("j-axis", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.7)

            # Set limits with padding
            min_x = min(state["points"][:, 0]) - 1
            max_x = max(state["points"][:, 0]) + 1
            min_y = min(state["points"][:, 1]) - 1
            max_y = max(state["points"][:, 1]) + 1

            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)

            plt.tight_layout()
            plt.show()


# Create a visualizer for a 10x10 iteration space with dependencies (1,0), (1,1), and (1,-1)
visualizer = LoopTransformationVisualizer(
    (5, 5), [(1, 1)], show_all=True
)

# Apply a skew transformation (i,j) → (i, i+j)
visualizer.custom_transformation(1, -1, 0, 1, name="Skew: (i,j) → (i-j, j)")

# Apply tiling with 2x2 tiles
visualizer.apply_tiling((1, 5), name="Tiling with 2x2 tiles")

# visualizer.custom_transformation(1, 1, 0, 1, name="Skew: (i,j) → (i+j, j)")

# Apply a transformation to the tiled space (optional)
# visualizer.skew_transformation(1, 0, name="Skew on Tiles: (ti,tj) → (ti+tj, tj)")

# Visualize the transformation history (optional)
# visualizer.visualize_history()
