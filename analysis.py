"""
Analysis and Visualization of Results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict


class ResultAnalyzer:
    """
    Analyzes experimental results and creates visualizations.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize analyzer with experimental data.

        Args:
            data: DataFrame with experimental results
        """
        self.data = data

    def compute_correlation_matrix(
        self,
        network_props: List[str] = None,
        spectral_metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Compute correlation matrix between network properties and spectral metrics.

        Args:
            network_props: List of network property column names
            spectral_metrics: List of spectral metric column names

        Returns:
            Correlation matrix DataFrame
        """
        if network_props is None:
            network_props = ['avg_degree', 'avg_betweenness_centrality', 'diameter', 'girth']

        if spectral_metrics is None:
            # Get all k0, k1, k2 metrics
            spectral_metrics = [col for col in self.data.columns if col.startswith('k')]

        # Select relevant columns
        selected_cols = network_props + spectral_metrics

        # Filter columns that exist in data
        selected_cols = [col for col in selected_cols if col in self.data.columns]

        # Compute correlation
        correlation = self.data[selected_cols].corr()

        # Extract the submatrix: network properties (rows) vs spectral metrics (columns)
        network_cols = [col for col in network_props if col in self.data.columns]
        spectral_cols = [col for col in spectral_metrics if col in self.data.columns]

        correlation_subset = correlation.loc[network_cols, spectral_cols]

        return correlation_subset

    def plot_correlation_heatmap(
        self,
        figsize: Tuple[int, int] = (16, 6),
        save_path: str = None
    ):
        """
        Plot correlation heatmap between network properties and spectral metrics.

        Args:
            figsize: Figure size
            save_path: Path to save the figure (optional)
        """
        correlation = self.compute_correlation_matrix()

        plt.figure(figsize=figsize)
        sns.heatmap(
            correlation,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        plt.title('Correlation: Network Properties vs Spectral Metrics', fontsize=14, fontweight='bold')
        plt.xlabel('Spectral Metrics', fontsize=12)
        plt.ylabel('Network Properties', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation heatmap saved to {save_path}")

        plt.show()

    def plot_eigenvalue_distributions(
        self,
        graph_type: str = 'erdos_renyi',
        k_values: List[int] = [0, 1, 2],
        figsize: Tuple[int, int] = (15, 5),
        save_path: str = None
    ):
        """
        Plot eigenvalue distributions for different Laplacians.

        Note: This requires storing individual eigenvalues, which we don't currently do.
        This is a placeholder for future implementation.

        Args:
            graph_type: Type of graph to plot
            k_values: List of k values to plot
            figsize: Figure size
            save_path: Path to save the figure (optional)
        """
        print("Note: Eigenvalue distribution plotting requires storing individual eigenvalues.")
        print("Currently, we only store aggregate spectral metrics.")
        print("To implement this, modify the experiment runner to store eigenvalues.")

    def plot_metric_vs_parameter(
        self,
        metric: str,
        parameter: str,
        graph_type: str = None,
        figsize: Tuple[int, int] = (10, 6),
        save_path: str = None
    ):
        """
        Plot a spectral metric vs a parameter (e.g., p for Erdős-Rényi).

        Args:
            metric: Name of the metric to plot
            parameter: Name of the parameter (x-axis)
            graph_type: Type of graph to filter (optional)
            figsize: Figure size
            save_path: Path to save the figure (optional)
        """
        data = self.data

        if graph_type:
            data = data[data['graph_type'] == graph_type]

        if metric not in data.columns or parameter not in data.columns:
            print(f"Error: Metric '{metric}' or parameter '{parameter}' not found in data.")
            return

        # Group by parameter and compute mean and std
        grouped = data.groupby(parameter)[metric].agg(['mean', 'std']).reset_index()

        plt.figure(figsize=figsize)
        plt.errorbar(
            grouped[parameter],
            grouped['mean'],
            yerr=grouped['std'],
            marker='o',
            linestyle='-',
            capsize=5,
            label=metric
        )
        plt.xlabel(parameter, fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.title(f'{metric} vs {parameter}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_network_property_vs_spectral_metric(
        self,
        network_prop: str,
        spectral_metric: str,
        graph_type: str = None,
        figsize: Tuple[int, int] = (8, 6),
        save_path: str = None
    ):
        """
        Scatter plot of network property vs spectral metric.

        Args:
            network_prop: Network property name
            spectral_metric: Spectral metric name
            graph_type: Type of graph to filter (optional)
            figsize: Figure size
            save_path: Path to save the figure (optional)
        """
        data = self.data

        if graph_type:
            data = data[data['graph_type'] == graph_type]

        if network_prop not in data.columns or spectral_metric not in data.columns:
            print(f"Error: Property '{network_prop}' or metric '{spectral_metric}' not found.")
            return

        # Remove infinite values
        data_clean = data[[network_prop, spectral_metric]].replace([np.inf, -np.inf], np.nan).dropna()

        plt.figure(figsize=figsize)
        plt.scatter(data_clean[network_prop], data_clean[spectral_metric], alpha=0.5)
        plt.xlabel(network_prop, fontsize=12)
        plt.ylabel(spectral_metric, fontsize=12)

        # Compute correlation
        corr = data_clean[network_prop].corr(data_clean[spectral_metric])
        plt.title(f'{spectral_metric} vs {network_prop}\n(Correlation: {corr:.3f})',
                  fontsize=12, fontweight='bold')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def generate_summary_statistics(self) -> pd.DataFrame:
        """
        Generate summary statistics for all metrics.

        Returns:
            DataFrame with summary statistics
        """
        # Get numeric columns only
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns

        # Compute summary statistics
        summary = self.data[numeric_cols].describe()

        return summary

    def save_results(self, filepath: str = 'results.csv'):
        """
        Save results to CSV file.

        Args:
            filepath: Path to save the CSV file
        """
        self.data.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")


def test_analysis():
    """Test analysis on dummy data."""
    # Create dummy data
    np.random.seed(42)
    n = 100

    data = pd.DataFrame({
        'graph_type': ['erdos_renyi'] * n,
        'p': np.random.uniform(0.1, 0.5, n),
        'avg_degree': np.random.uniform(5, 20, n),
        'diameter': np.random.uniform(2, 10, n),
        'girth': np.random.uniform(3, 8, n),
        'avg_betweenness_centrality': np.random.uniform(0, 0.5, n),
        'k0_nullity': np.random.randint(0, 5, n),
        'k0_trace': np.random.uniform(50, 200, n),
        'k1_nullity': np.random.randint(0, 10, n),
        'k1_trace': np.random.uniform(100, 400, n),
        'k2_nullity': np.random.randint(0, 3, n),
        'k2_trace': np.random.uniform(0, 50, n)
    })

    analyzer = ResultAnalyzer(data)

    print("Testing correlation matrix computation:")
    corr = analyzer.compute_correlation_matrix()
    print(corr)

    print("\nTesting correlation heatmap:")
    analyzer.plot_correlation_heatmap()


if __name__ == "__main__":
    test_analysis()
