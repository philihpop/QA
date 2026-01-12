"""
Simplicial Complex and Combinatorial Laplacian Construction
"""

import numpy as np
import networkx as nx
from itertools import combinations
from typing import List, Dict, Tuple, Set


class SimplicialComplex:
    """
    Constructs a clique complex from a graph and computes combinatorial Laplacians.
    """

    def __init__(self, graph: nx.Graph, max_k: int = 2):
        """
        Initialize simplicial complex from a graph.

        Args:
            graph: NetworkX graph
            max_k: Maximum dimension of simplices to compute (0=nodes, 1=edges, 2=triangles)
        """
        self.graph = graph
        self.max_k = max_k
        self.simplices = {}  # {k: list of k-simplices}
        self.simplex_to_idx = {}  # {k: {simplex: index}}

        self._build_clique_complex()

    def _build_clique_complex(self):
        """
        Build the clique complex by finding all cliques up to size max_k+1.
        A k-simplex corresponds to a (k+1)-clique.
        """
        # 0-simplices are just nodes
        self.simplices[0] = [frozenset([node]) for node in self.graph.nodes()]
        self.simplex_to_idx[0] = {s: i for i, s in enumerate(self.simplices[0])}

        # 1-simplices are edges
        self.simplices[1] = [frozenset(edge) for edge in self.graph.edges()]
        self.simplex_to_idx[1] = {s: i for i, s in enumerate(self.simplices[1])}

        # Higher-order simplices: find all cliques of size k+1
        if self.max_k >= 2:
            # Find all cliques
            cliques = list(nx.find_cliques(self.graph))

            for k in range(2, self.max_k + 1):
                self.simplices[k] = []
                # Extract k-simplices from cliques
                for clique in cliques:
                    if len(clique) >= k + 1:
                        # Generate all (k+1)-subsets of the clique
                        for subset in combinations(clique, k + 1):
                            simplex = frozenset(subset)
                            if simplex not in self.simplices[k]:
                                self.simplices[k].append(simplex)

                self.simplex_to_idx[k] = {s: i for i, s in enumerate(self.simplices[k])}

    def get_boundary_operator(self, k: int) -> np.ndarray:
        """
        Compute the boundary operator ∂_k: C_k -> C_{k-1}

        Args:
            k: Dimension of the boundary operator

        Returns:
            Boundary matrix of shape (n_{k-1}, n_k)
        """
        if k == 0:
            # Boundary of 0-simplices is empty
            return np.zeros((0, len(self.simplices[0])))

        if k not in self.simplices or k - 1 not in self.simplices:
            raise ValueError(f"Simplices of dimension {k} or {k-1} not computed")

        n_k_minus_1 = len(self.simplices[k - 1])
        n_k = len(self.simplices[k])

        boundary = np.zeros((n_k_minus_1, n_k))

        for j, simplex in enumerate(self.simplices[k]):
            # For each k-simplex, find its (k-1)-faces
            simplex_list = list(simplex)
            for i, vertex in enumerate(simplex_list):
                # Remove vertex to get (k-1)-face
                face = frozenset(simplex_list[:i] + simplex_list[i+1:])
                if face in self.simplex_to_idx[k - 1]:
                    face_idx = self.simplex_to_idx[k - 1][face]
                    # Alternating sign based on position
                    boundary[face_idx, j] = (-1) ** i

        return boundary

    def get_combinatorial_laplacian(self, k: int) -> np.ndarray:
        """
        Compute the k-th combinatorial Laplacian: Δ^k = ∂_{k+1}∂_{k+1}^T + ∂_k^T∂_k

        Args:
            k: Dimension of the Laplacian

        Returns:
            Laplacian matrix of shape (n_k, n_k)
        """
        n_k = len(self.simplices[k])

        # Initialize Laplacian
        laplacian = np.zeros((n_k, n_k))

        # Add ∂_k^T∂_k term (down Laplacian)
        if k >= 1:
            boundary_k = self.get_boundary_operator(k)
            laplacian += boundary_k.T @ boundary_k

        # Add ∂_{k+1}∂_{k+1}^T term (up Laplacian)
        if k + 1 in self.simplices and len(self.simplices[k + 1]) > 0:
            boundary_k_plus_1 = self.get_boundary_operator(k + 1)
            laplacian += boundary_k_plus_1 @ boundary_k_plus_1.T

        return laplacian

    def get_eigenvalues(self, k: int) -> np.ndarray:
        """
        Compute eigenvalues of the k-th combinatorial Laplacian.

        Args:
            k: Dimension of the Laplacian

        Returns:
            Array of eigenvalues sorted in ascending order
        """
        laplacian = self.get_combinatorial_laplacian(k)

        if laplacian.shape[0] == 0:
            return np.array([])

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(laplacian)

        # Sort in ascending order and remove numerical noise near zero
        eigenvalues = np.sort(eigenvalues)
        eigenvalues[np.abs(eigenvalues) < 1e-10] = 0.0

        return eigenvalues

    def get_all_eigenvalues(self) -> Dict[int, np.ndarray]:
        """
        Compute eigenvalues for all Laplacians up to max_k.

        Returns:
            Dictionary mapping k to eigenvalues of Δ^k
        """
        return {k: self.get_eigenvalues(k) for k in range(self.max_k + 1)}


def test_simplicial_complex():
    """Test the simplicial complex construction on a simple triangle graph."""
    # Create a triangle graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    sc = SimplicialComplex(G, max_k=2)

    print("Testing Simplicial Complex on a triangle graph:")
    print(f"0-simplices (nodes): {len(sc.simplices[0])}")
    print(f"1-simplices (edges): {len(sc.simplices[1])}")
    print(f"2-simplices (triangles): {len(sc.simplices[2])}")

    print("\nBoundary operator ∂_1:")
    print(sc.get_boundary_operator(1))

    print("\nLaplacian Δ^0:")
    print(sc.get_combinatorial_laplacian(0))

    print("\nEigenvalues of Δ^0:")
    print(sc.get_eigenvalues(0))

    print("\nLaplacian Δ^1:")
    print(sc.get_combinatorial_laplacian(1))

    print("\nEigenvalues of Δ^1:")
    print(sc.get_eigenvalues(1))


if __name__ == "__main__":
    test_simplicial_complex()
