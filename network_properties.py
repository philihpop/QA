"""
Network Property Calculations for Graphs
"""

import numpy as np
import networkx as nx
from typing import Dict


class NetworkProperties:
    """
    Computes standard network properties for graphs.
    """

    @staticmethod
    def compute_average_degree(graph: nx.Graph) -> float:
        """
        Compute average degree of the graph.

        Args:
            graph: NetworkX graph

        Returns:
            Average degree
        """
        if len(graph.nodes()) == 0:
            return 0.0
        degrees = [d for n, d in graph.degree()]
        return float(np.mean(degrees))

    @staticmethod
    def compute_avg_betweenness_centrality(graph: nx.Graph) -> float:
        """
        Compute average betweenness centrality.

        Args:
            graph: NetworkX graph

        Returns:
            Average betweenness centrality
        """
        if len(graph.nodes()) == 0:
            return 0.0

        centrality = nx.betweenness_centrality(graph)
        return float(np.mean(list(centrality.values())))

    @staticmethod
    def compute_diameter(graph: nx.Graph) -> float:
        """
        Compute diameter of the graph (longest shortest path).
        Returns infinity if graph is disconnected.

        Args:
            graph: NetworkX graph

        Returns:
            Diameter (or inf if disconnected)
        """
        if len(graph.nodes()) == 0:
            return 0.0

        if not nx.is_connected(graph):
            # For disconnected graphs, return the maximum diameter of connected components
            components = list(nx.connected_components(graph))
            if len(components) == 0:
                return 0.0
            diameters = []
            for comp in components:
                subgraph = graph.subgraph(comp)
                if len(subgraph.nodes()) > 1:
                    diameters.append(nx.diameter(subgraph))
            return float(max(diameters)) if diameters else 0.0

        return float(nx.diameter(graph))

    @staticmethod
    def compute_girth(graph: nx.Graph) -> float:
        """
        Compute girth (length of shortest cycle) of the graph.
        Returns infinity if the graph is acyclic.

        Args:
            graph: NetworkX graph

        Returns:
            Girth (or inf if acyclic)
        """
        if len(graph.nodes()) == 0:
            return np.inf

        # Find minimum cycle basis
        try:
            cycles = nx.minimum_cycle_basis(graph)
            if len(cycles) == 0:
                return np.inf
            girth = min(len(cycle) for cycle in cycles)
            return float(girth)
        except:
            return np.inf

    @staticmethod
    def compute_all_properties(graph: nx.Graph) -> Dict[str, float]:
        """
        Compute all network properties.

        Args:
            graph: NetworkX graph

        Returns:
            Dictionary of all network properties
        """
        return {
            "avg_degree": NetworkProperties.compute_average_degree(graph),
            "avg_betweenness_centrality": NetworkProperties.compute_avg_betweenness_centrality(graph),
            "diameter": NetworkProperties.compute_diameter(graph),
            "girth": NetworkProperties.compute_girth(graph),
            "num_nodes": len(graph.nodes()),
            "num_edges": len(graph.edges())
        }


def test_network_properties():
    """Test network properties on example graphs."""
    # Create a cycle graph
    G = nx.cycle_graph(5)

    print("Testing Network Properties on a 5-cycle:")
    print(f"Graph: {G.edges()}")
    print()

    properties = NetworkProperties.compute_all_properties(G)

    for name, value in properties.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    test_network_properties()
