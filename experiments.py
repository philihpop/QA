"""
Experiment Generation and Data Collection
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
import pandas as pd
from simplicial_complex import SimplicialComplex
from spectral_metrics import SpectralMetrics
from network_properties import NetworkProperties


class ExperimentRunner:
    """
    Runs experiments on different graph types and collects data.
    """

    def __init__(self, max_k: int = 2):
        """
        Initialize experiment runner.

        Args:
            max_k: Maximum dimension of simplices to compute
        """
        self.max_k = max_k

    def analyze_single_graph(self, graph: nx.Graph) -> Dict:
        """
        Analyze a single graph: compute network properties and spectral metrics.

        Args:
            graph: NetworkX graph

        Returns:
            Dictionary containing all properties and metrics
        """
        result = {}

        # Compute network properties
        net_props = NetworkProperties.compute_all_properties(graph)
        result.update(net_props)

        # Build simplicial complex and compute eigenvalues
        sc = SimplicialComplex(graph, max_k=self.max_k)

        # Compute spectral metrics for each dimension k
        for k in range(self.max_k + 1):
            eigenvalues = sc.get_eigenvalues(k)
            metrics = SpectralMetrics.compute_all_metrics(eigenvalues)

            # Prefix metrics with k value
            for metric_name, metric_value in metrics.items():
                result[f"k{k}_{metric_name}"] = metric_value

        return result

    def run_erdos_renyi_experiment(
        self,
        n_nodes: int = 50,
        p_values: List[float] = None,
        n_instances: int = 30
    ) -> pd.DataFrame:
        """
        Run Erdős-Rényi graph experiment.

        Args:
            n_nodes: Number of nodes
            p_values: List of edge probabilities to test
            n_instances: Number of instances per p-value

        Returns:
            DataFrame with results
        """
        if p_values is None:
            p_values = np.arange(0.05, 0.55, 0.05)

        results = []

        print(f"Running Erdős-Rényi experiment: n={n_nodes}, p={p_values}, instances={n_instances}")

        for p in p_values:
            print(f"  Testing p={p:.2f}...")
            for instance in range(n_instances):
                # Generate random graph
                G = nx.erdos_renyi_graph(n_nodes, p, seed=instance)

                # Analyze
                result = self.analyze_single_graph(G)
                result['graph_type'] = 'erdos_renyi'
                result['p'] = p
                result['instance'] = instance

                results.append(result)

        return pd.DataFrame(results)

    def run_karate_club_filtration(self, n_steps: int = 20) -> pd.DataFrame:
        """
        Run Zachary Karate Club with edge-weight filtration.

        Args:
            n_steps: Number of filtration steps

        Returns:
            DataFrame with results
        """
        results = []

        print("Running Zachary Karate Club filtration experiment...")

        # Load Karate Club graph
        G = nx.karate_club_graph()

        # Assign random weights to edges
        np.random.seed(42)
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.uniform(0, 1)

        # Create filtration
        epsilon_values = np.linspace(0.1, 1.0, n_steps)

        for epsilon in epsilon_values:
            print(f"  Filtration threshold ε={epsilon:.2f}...")

            # Build filtered graph
            G_filtered = nx.Graph()
            G_filtered.add_nodes_from(G.nodes())

            for u, v, data in G.edges(data=True):
                if data['weight'] <= epsilon:
                    G_filtered.add_edge(u, v)

            # Analyze
            result = self.analyze_single_graph(G_filtered)
            result['graph_type'] = 'karate_club'
            result['epsilon'] = epsilon

            results.append(result)

        return pd.DataFrame(results)

    def run_weighted_filtration_experiment(
        self,
        n_nodes: int = 50,
        n_instances: int = 10,
        n_steps: int = 15
    ) -> pd.DataFrame:
        """
        Run general edge-weight filtration experiment on random graphs.

        Args:
            n_nodes: Number of nodes
            n_instances: Number of random graph instances
            n_steps: Number of filtration steps per instance

        Returns:
            DataFrame with results
        """
        results = []

        print(f"Running weighted filtration experiment: n={n_nodes}, instances={n_instances}, steps={n_steps}")

        for instance in range(n_instances):
            print(f"  Instance {instance + 1}/{n_instances}...")

            # Generate a random graph with enough edges for interesting filtration
            G = nx.erdos_renyi_graph(n_nodes, p=0.3, seed=instance)

            # Assign random weights
            for u, v in G.edges():
                G[u][v]['weight'] = np.random.uniform(0, 1)

            # Create filtration
            epsilon_values = np.linspace(0.1, 1.0, n_steps)

            for epsilon in epsilon_values:
                # Build filtered graph
                G_filtered = nx.Graph()
                G_filtered.add_nodes_from(G.nodes())

                for u, v, data in G.edges(data=True):
                    if data['weight'] <= epsilon:
                        G_filtered.add_edge(u, v)

                # Analyze
                result = self.analyze_single_graph(G_filtered)
                result['graph_type'] = 'weighted_filtration'
                result['epsilon'] = epsilon
                result['instance'] = instance

                results.append(result)

        return pd.DataFrame(results)

    def run_all_experiments(self) -> pd.DataFrame:
        """
        Run all three experiments and combine results.

        Returns:
            Combined DataFrame with all results
        """
        print("=" * 60)
        print("Starting all experiments...")
        print("=" * 60)

        # Experiment 1: Erdős-Rényi
        df_er = self.run_erdos_renyi_experiment(
            n_nodes=75,
            p_values=np.arange(0.05, 0.55, 0.05),
            n_instances=30
        )

        print()

        # Experiment 2: Karate Club
        df_kc = self.run_karate_club_filtration(n_steps=20)

        print()

        # Experiment 3: Weighted Filtration
        df_wf = self.run_weighted_filtration_experiment(
            n_nodes=50,
            n_instances=10,
            n_steps=15
        )

        print()
        print("=" * 60)
        print("All experiments completed!")
        print("=" * 60)

        # Combine all results
        df_all = pd.concat([df_er, df_kc, df_wf], ignore_index=True)

        return df_all


def test_experiments():
    """Test experiment runner on a small example."""
    runner = ExperimentRunner(max_k=2)

    # Test on a small Erdős-Rényi graph
    print("Testing on small Erdős-Rényi graphs:")
    df = runner.run_erdos_renyi_experiment(n_nodes=20, p_values=[0.1, 0.3], n_instances=3)

    print("\nResults shape:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())


if __name__ == "__main__":
    test_experiments()
