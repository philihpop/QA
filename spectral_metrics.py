"""
Spectral Metrics Computation for Combinatorial Laplacians
"""

import numpy as np
from typing import Dict, List


class SpectralMetrics:
    """
    Computes the six spectral metrics from eigenvalues of combinatorial Laplacians.
    """

    @staticmethod
    def compute_nullity(eigenvalues: np.ndarray, threshold: float = 1e-10) -> int:
        """
        Compute nullity: number of zero eigenvalues (Betti number).

        Args:
            eigenvalues: Array of eigenvalues
            threshold: Threshold below which eigenvalues are considered zero

        Returns:
            Nullity (number of zero eigenvalues)
        """
        return int(np.sum(np.abs(eigenvalues) < threshold))

    @staticmethod
    def compute_low_lying_density(eigenvalues: np.ndarray, bound: float) -> int:
        """
        Compute low-lying spectral density: number of eigenvalues below bound b.

        Args:
            eigenvalues: Array of eigenvalues
            bound: Upper bound for counting eigenvalues

        Returns:
            Number of eigenvalues < bound
        """
        return int(np.sum(eigenvalues < bound))

    @staticmethod
    def compute_trace(eigenvalues: np.ndarray) -> float:
        """
        Compute trace: sum of all eigenvalues.

        Args:
            eigenvalues: Array of eigenvalues

        Returns:
            Trace (sum of eigenvalues)
        """
        return float(np.sum(eigenvalues))

    @staticmethod
    def compute_spectral_moment(eigenvalues: np.ndarray, ell: int) -> float:
        """
        Compute ℓ-th spectral moment: sum of λ_j^ℓ.

        Args:
            eigenvalues: Array of eigenvalues
            ell: Power for the moment

        Returns:
            ℓ-th spectral moment
        """
        return float(np.sum(eigenvalues ** ell))

    @staticmethod
    def compute_quasi_wiener_index(eigenvalues: np.ndarray, threshold: float = 1e-10) -> float:
        """
        Compute quasi-Wiener index: sum of (rank + 1) / λ_j for non-zero eigenvalues.

        Args:
            eigenvalues: Array of eigenvalues
            threshold: Threshold below which eigenvalues are considered zero

        Returns:
            Quasi-Wiener index
        """
        # Filter out zero eigenvalues
        nonzero_eigenvalues = eigenvalues[np.abs(eigenvalues) >= threshold]

        if len(nonzero_eigenvalues) == 0:
            return 0.0

        rank = len(eigenvalues) - np.sum(np.abs(eigenvalues) < threshold)

        return float(np.sum((rank + 1) / nonzero_eigenvalues))

    @staticmethod
    def compute_spanning_tree_number(eigenvalues: np.ndarray, threshold: float = 1e-10) -> float:
        """
        Compute spanning-tree number: log((1/(rank+1)) * product of non-zero eigenvalues).

        Args:
            eigenvalues: Array of eigenvalues
            threshold: Threshold below which eigenvalues are considered zero

        Returns:
            Spanning-tree number (log scale)
        """
        # Filter out zero eigenvalues
        nonzero_eigenvalues = eigenvalues[np.abs(eigenvalues) >= threshold]

        if len(nonzero_eigenvalues) == 0:
            return -np.inf

        rank = len(eigenvalues) - np.sum(np.abs(eigenvalues) < threshold)

        # Compute product in log space to avoid numerical overflow
        log_product = np.sum(np.log(nonzero_eigenvalues))
        log_result = log_product - np.log(rank + 1)

        return float(log_result)

    @staticmethod
    def compute_all_metrics(
        eigenvalues: np.ndarray,
        low_lying_bound: float = 1.0,
        moment_power: int = 2
    ) -> Dict[str, float]:
        """
        Compute all six spectral metrics.

        Args:
            eigenvalues: Array of eigenvalues
            low_lying_bound: Bound for low-lying spectral density
            moment_power: Power ℓ for spectral moment

        Returns:
            Dictionary of all metrics
        """
        if len(eigenvalues) == 0:
            return {
                "nullity": 0,
                "low_lying_density": 0,
                "trace": 0.0,
                f"spectral_moment_{moment_power}": 0.0,
                "quasi_wiener_index": 0.0,
                "spanning_tree_number": -np.inf
            }

        return {
            "nullity": SpectralMetrics.compute_nullity(eigenvalues),
            "low_lying_density": SpectralMetrics.compute_low_lying_density(eigenvalues, low_lying_bound),
            "trace": SpectralMetrics.compute_trace(eigenvalues),
            f"spectral_moment_{moment_power}": SpectralMetrics.compute_spectral_moment(eigenvalues, moment_power),
            "quasi_wiener_index": SpectralMetrics.compute_quasi_wiener_index(eigenvalues),
            "spanning_tree_number": SpectralMetrics.compute_spanning_tree_number(eigenvalues)
        }


def test_spectral_metrics():
    """Test spectral metrics on example eigenvalues."""
    eigenvalues = np.array([0.0, 0.0, 1.0, 2.0, 3.5, 5.0])

    print("Testing Spectral Metrics:")
    print(f"Eigenvalues: {eigenvalues}")
    print()

    metrics = SpectralMetrics.compute_all_metrics(eigenvalues, low_lying_bound=2.5, moment_power=2)

    for name, value in metrics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    test_spectral_metrics()
