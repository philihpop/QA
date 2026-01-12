# Topological Data Analysis: Spectral Fingerprinting of Networks

This repository contains the implementation for Project 1: analyzing complex networks using spectral properties of combinatorial Laplacians.

## Overview

The project implements:
1. **Simplicial complex construction** from graphs (clique complexes)
2. **Combinatorial Laplacian computation** for dimensions k=0,1,2
3. **Six spectral metrics** from Laplacian eigenvalues
4. **Network property calculations** (degree, centrality, diameter, girth)
5. **Three experimental setups**:
   - Erdős-Rényi random graphs
   - Zachary's Karate Club with filtration
   - Weighted graph filtrations
6. **Correlation analysis** between spectral and network properties

## Project Structure

```
.
├── simplicial_complex.py    # Simplicial complex and Laplacian construction
├── spectral_metrics.py       # Six spectral metrics computation
├── network_properties.py     # Network property calculations
├── experiments.py            # Experiment runners for three graph types
├── analysis.py              # Correlation analysis and visualization
├── main.py                  # Main script to run everything
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Requirements:
- networkx >= 3.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- pandas >= 2.0.0

## Usage

### Running All Experiments

To run all three experiments and generate analysis:

```bash
python main.py
```

This will:
1. Generate Erdős-Rényi graphs (n=75 nodes, p ∈ [0.05, 0.5], 30 instances per p)
2. Analyze Zachary's Karate Club with 20 filtration steps
3. Run weighted filtration on 10 random graph instances
4. Compute correlation matrices
5. Generate visualizations
6. Save all results to the `outputs/` directory

### Loading Existing Results

If you've already run experiments and want to re-analyze:

```bash
python main.py --load --data-file experiment_results.csv
```

### Testing Individual Components

Test each module independently:

```bash
# Test simplicial complex construction
python simplicial_complex.py

# Test spectral metrics
python spectral_metrics.py

# Test network properties
python network_properties.py

# Test experiment runner (small test)
python experiments.py

# Test analysis (with dummy data)
python analysis.py
```

## Implementation Details

### Simplicial Complex Construction

The `SimplicialComplex` class builds a clique complex from a NetworkX graph:
- **0-simplices**: Nodes
- **1-simplices**: Edges
- **2-simplices**: Triangles (3-cliques)

### Combinatorial Laplacian

For dimension k, the Laplacian is computed as:

```
Δ^k = ∂_{k+1} ∂_{k+1}^T + ∂_k^T ∂_k
```

where ∂_k is the boundary operator mapping k-simplices to (k-1)-simplices.

### Six Spectral Metrics

Given eigenvalues {λ_j} of Δ^k:

1. **Nullity**: Number of zero eigenvalues (Betti-k number)
2. **Low-lying Density**: Number of eigenvalues below threshold b
3. **Trace**: Sum of eigenvalues
4. **Spectral Moment**: Sum of λ_j^ℓ (default ℓ=2)
5. **Quasi-Wiener Index**: Sum of (rank+1)/λ_j for non-zero λ_j
6. **Spanning-tree Number**: log((1/(rank+1)) × product of non-zero λ_j)

### Network Properties

Four key network properties are computed:
1. **Average Degree**: Mean node degree
2. **Average Betweenness Centrality**: Mean betweenness across nodes
3. **Diameter**: Longest shortest path
4. **Girth**: Length of shortest cycle

## Experiments

### Experiment 1: Erdős-Rényi Graphs

- **Nodes**: 75
- **Edge probability**: p ∈ [0.05, 0.5] in steps of 0.05
- **Instances**: 30 per p-value
- **Goal**: Observe phase transition from disconnected to connected

### Experiment 2: Zachary's Karate Club

- **Graph**: Karate Club (34 nodes)
- **Method**: Edge-weight filtration with 20 steps
- **Goal**: Detect community structure via spectral properties

### Experiment 3: Weighted Filtrations

- **Nodes**: 50
- **Instances**: 10 random graphs
- **Filtration steps**: 15 per instance
- **Goal**: Track Betti number evolution

## Output Files

After running `main.py`, the following files are generated in `outputs/`:

- `experiment_results.csv`: Raw data for all experiments
- `correlation_matrix.csv`: Correlation between network and spectral properties
- `correlation_heatmap.png`: Heatmap visualization
- `betti0_vs_p.png`: Betti-0 number vs edge probability
- `betti1_vs_p.png`: Betti-1 number vs edge probability
- `trace_vs_degree.png`: Trace(Δ^0) vs average degree
- `quasi_wiener_vs_diameter.png`: Quasi-Wiener index vs diameter

## Expected Correlations

Based on the plan, we expect:

| Network Property | Spectral Correlate |
|-----------------|-------------------|
| Connectivity | Nullity of Δ^0 (Betti-0) |
| Cycle Density | Nullity of Δ^1 (Betti-1) |
| Network Efficiency | Quasi-Wiener Index |
| Reliability | Spanning-tree Number |
| Average Degree | Trace of Δ^0 |
| Diameter | Quasi-Wiener Index |
| Girth | Spectral Gap of Δ^1 |

## Computational Complexity

- **Clique finding**: O(3^(n/3)) worst case, but efficient for sparse graphs
- **Eigenvalue decomposition**: O(m^3) where m is the number of k-simplices
- **Boundary operator construction**: O(mk) where m is simplices, k is dimension

For the experiments:
- Erdős-Rényi (n=75, 300 instances): ~2-5 minutes
- Karate Club filtration (20 steps): ~10-30 seconds
- Weighted filtrations (10×15 steps): ~1-2 minutes

Total runtime: ~5-10 minutes on a standard laptop.

## References

1. Quantum algorithm for topological data analysis: https://arxiv.org/abs/2005.02607
2. Complex network analysis using combinatorial Laplacian: https://link.springer.com/content/pdf/10.1140%2Fepjst%2Fe2012-01655-6.pdf
3. Graph filtrations: https://pages.stat.wisc.edu/~mchung/papers/chung.2019.NN

## Notes for Report

Key findings to include in the report:

1. **Betti-0 (k0_nullity)** captures the number of connected components
   - Should decrease as p increases in Erdős-Rényi
   - Should be 1 for connected graphs

2. **Betti-1 (k1_nullity)** captures the number of independent cycles
   - Should increase then plateau as p increases
   - Indicates topological complexity

3. **Trace correlations** with average degree
   - Trace(Δ^0) = sum of node degrees (for k=0)
   - Strong positive correlation expected

4. **Quasi-Wiener Index** relates to graph efficiency
   - Lower values = more efficient network
   - Should correlate with diameter

5. **Community detection** in Karate Club
   - Watch for jumps in Betti numbers at critical filtration levels
   - These indicate structural transitions

## License

Academic project for educational purposes.
