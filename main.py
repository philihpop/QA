import argparse
import os
import numpy as np
from experiments import ExperimentRunner
from analysis import ResultAnalyzer


def main(
    run_experiments: bool = True,
    load_existing: bool = False,
    data_file: str = 'experiment_results.csv',
    output_dir: str = 'outputs'
):
    """
    Main function to run experiments and analysis with optimized parameters.

    Args:
        run_experiments: Whether to run experiments or load existing data
        load_existing: Whether to load existing results file
        data_file: Path to data file
        output_dir: Directory to save outputs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Run experiments or load existing data
    if run_experiments and not load_existing:
        print("Running all experiments with optimized parameters...")
        print("Estimated time: 3-5 minutes\n")

        runner = ExperimentRunner(max_k=2)

        print("=" * 60)
        print("Starting all experiments...")
        print("=" * 60)

        # Experiment 1: Erdős-Rényi (OPTIMIZED)
        # Reduced: 75→50 nodes, 30→20 instances, p max 0.50→0.35
        # Still captures phase transition, much faster
        print("\n[1/3] Erdős-Rényi Graphs")
        print("-" * 60)
        df_er = runner.run_erdos_renyi_experiment(
            n_nodes=50,                              # Reduced from 75
            p_values=np.arange(0.05, 0.40, 0.05),   # Reduced from 0.55
            n_instances=20                           # Reduced from 30
        )
        print(f"✓ Generated {len(df_er)} Erdős-Rényi samples")

        # Experiment 2: Karate Club (UNCHANGED)
        # Already small and fast (~30 seconds)
        print("\n[2/3] Zachary's Karate Club")
        print("-" * 60)
        df_kc = runner.run_karate_club_filtration(n_steps=20)
        print(f"✓ Generated {len(df_kc)} Karate Club filtration samples")

        # Experiment 3: Weighted Filtration (OPTIMIZED)
        # Reduced: 50→40 nodes, 10→5 instances, 15→10 steps
        print("\n[3/3] Weighted Graph Filtrations")
        print("-" * 60)
        df_wf = runner.run_weighted_filtration_experiment(
            n_nodes=40,        # Reduced from 50
            n_instances=5,     # Reduced from 10
            n_steps=10         # Reduced from 15
        )
        print(f"✓ Generated {len(df_wf)} weighted filtration samples")

        print("\n" + "=" * 60)
        print("All experiments completed!")
        print("=" * 60)

        # Combine all results
        import pandas as pd
        data = pd.concat([df_er, df_kc, df_wf], ignore_index=True)

        print(f"\nTotal samples: {len(data)}")
        print(f"  - Erdős-Rényi: {len(df_er)}")
        print(f"  - Karate Club: {len(df_kc)}")
        print(f"  - Weighted Filtration: {len(df_wf)}")

        # Save results
        data.to_csv(data_file, index=False)
        print(f"\nExperiment results saved to {data_file}")

    elif load_existing:
        print(f"Loading existing results from {data_file}...")
        import pandas as pd
        data = pd.read_csv(data_file)
        print(f"Loaded {len(data)} rows of data")

    else:
        print("Error: Must either run experiments or load existing data.")
        return

    # Create analyzer
    print("\n" + "=" * 60)
    print("Starting analysis...")
    print("=" * 60 + "\n")

    analyzer = ResultAnalyzer(data)

    # Generate summary statistics
    print("Summary Statistics:")
    print(analyzer.generate_summary_statistics())
    print()

    # Compute and display correlation matrix
    print("Computing correlation matrix...")
    correlation = analyzer.compute_correlation_matrix()
    print("\nCorrelation Matrix (Network Properties vs Spectral Metrics):")
    print(correlation)
    print()

    # Save correlation matrix
    corr_file = os.path.join(output_dir, 'correlation_matrix.csv')
    correlation.to_csv(corr_file)
    print(f"Correlation matrix saved to {corr_file}\n")

    # Plot correlation heatmap
    print("Generating correlation heatmap...")
    heatmap_file = os.path.join(output_dir, 'correlation_heatmap.png')
    analyzer.plot_correlation_heatmap(save_path=heatmap_file)
    print()

    # Plot some interesting relationships
    print("Generating additional plots...")

    # Plot nullity vs p for Erdős-Rényi (Betti numbers)
    print("  - Plotting Betti-0 vs edge probability...")
    analyzer.plot_metric_vs_parameter(
        metric='k0_nullity',
        parameter='p',
        graph_type='erdos_renyi',
        save_path=os.path.join(output_dir, 'betti0_vs_p.png')
    )

    print("  - Plotting Betti-1 vs edge probability...")
    analyzer.plot_metric_vs_parameter(
        metric='k1_nullity',
        parameter='p',
        graph_type='erdos_renyi',
        save_path=os.path.join(output_dir, 'betti1_vs_p.png')
    )

    # Plot trace vs average degree
    print("  - Plotting trace(Δ^0) vs average degree...")
    analyzer.plot_network_property_vs_spectral_metric(
        network_prop='avg_degree',
        spectral_metric='k0_trace',
        save_path=os.path.join(output_dir, 'trace_vs_degree.png')
    )

    # Plot quasi-Wiener index vs diameter
    print("  - Plotting quasi-Wiener index vs diameter...")
    analyzer.plot_network_property_vs_spectral_metric(
        network_prop='diameter',
        spectral_metric='k0_quasi_wiener_index',
        save_path=os.path.join(output_dir, 'quasi_wiener_vs_diameter.png')
    )

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print(f"\nAll outputs saved to '{output_dir}/' directory")
    print(f"  - {data_file}: Raw experimental data")
    print(f"  - {corr_file}: Correlation matrix")
    print(f"  - correlation_heatmap.png: Correlation heatmap visualization")
    print(f"  - *.png: Additional analysis plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run Topological Data Analysis experiments and analysis (OPTIMIZED)'
    )

    parser.add_argument(
        '--load',
        action='store_true',
        help='Load existing results instead of running experiments'
    )

    parser.add_argument(
        '--data-file',
        type=str,
        default='experiment_results.csv',
        help='Path to data file (default: experiment_results.csv)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Directory to save outputs (default: outputs)'
    )

    args = parser.parse_args()

    main(
        run_experiments=not args.load,
        load_existing=args.load,
        data_file=args.data_file,
        output_dir=args.output_dir
    )
