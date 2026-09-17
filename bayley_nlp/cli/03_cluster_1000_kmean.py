
# --- UTF-8 stdout/stderr: this script prints Unicode (arrows, checkmarks)
# which crashes under Windows' default cp1252 encoding; force UTF-8. ---
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")
del _stream

#!/usr/bin/env python
"""
Step 3: Clustering - Multiple Strategy Options

Provides flexible clustering approaches for discovering item factors:
- kmeans_consensus: KMeans with consensus (1000 runs) - stable, reproducible
- hdbscan: Density-based clustering - model-free discovery
- kmeans_multi_init: KMeans with high n_init (future)
- similarity: Item-factor similarity-based (future)

Usage:
    # Default: KMeans consensus with 1000 runs
    python 03_cluster_1000_kmean.py
    
    # Specify strategy
    python 03_cluster_1000_kmean.py --strategy hdbscan
    python 03_cluster_1000_kmean.py --strategy kmeans_consensus --n-clusters 7
    
    # HDBSCAN with custom parameters
    python 03_cluster_1000_kmean.py --strategy hdbscan --min-cluster-size 5

Output (saved to data/outputs/experiments/{model}_analysis/03_clustering/):
    - cluster_assignments.csv    # Cluster assignments (compatible with 04_labeling.py)
    - umap_2d_clusters.png       # 2D UMAP visualization
    - Additional strategy-specific outputs (see documentation)
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress sklearn deprecation warnings from dependencies
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
# Suppress UMAP n_jobs warning when using random_state
warnings.filterwarnings('ignore', message='n_jobs value.*overridden.*random_state', module='umap')

from bayley_nlp.core import ensure_dir
from bayley_nlp.core.clustering import (
    KMeansConsensusStrategy,
    KMeansSimpleStrategy,
    HDBSCANStrategy,
    SimilarityClusteringStrategy,
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Perform clustering with multiple strategy options",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--strategy",
        type=str,
        default="kmeans_consensus",
        choices=["kmeans_consensus", "kmeans_simple", "hdbscan", "kmeans_multi_init", "similarity"],
        help="Clustering strategy to use",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="all-mpnet-base-v2",
        choices=["all-mpnet-base-v2", "all-roberta-large-v1"],
        help="Model used for embeddings",
    )
    
    # KMeans-specific parameters
    ap.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="Number of clusters for KMeans (default: 5)",
    )
    ap.add_argument(
        "--n-iterations",
        type=int,
        default=1000,
        help="Number of iterations for consensus strategies (default: 1000)",
    )
    ap.add_argument(
        "--consensus-linkage",
        type=str,
        default="average",
        choices=["average", "complete", "single"],
        help="Linkage method for consensus hierarchical clustering",
    )
    
    # HDBSCAN-specific parameters
    ap.add_argument(
        "--min-cluster-size",
        type=str,
        default="2,3,4,5,6",
        help="Comma-separated min cluster sizes to try (HDBSCAN)",
    )
    ap.add_argument(
        "--min-samples",
        type=str,
        default="none,1,2",
        help="Comma-separated min samples (use 'none' for None) (HDBSCAN)",
    )
    
    # Dimensionality reduction parameters (common)
    ap.add_argument(
        "--dr-method",
        type=str,
        default="umap",
        choices=["umap", "tsne", "pca", "isomap", "spectral", "mds"],
        help="Dimensionality reduction method (default: umap)",
    )
    ap.add_argument(
        "--no-dr",
        action="store_true",
        help="Skip dimensionality reduction and cluster directly on embeddings (high-dimensional)",
    )
    ap.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=10,
        help="n_neighbors parameter for UMAP/Isomap/Spectral",
    )
    ap.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.0,
        help="min_dist parameter for UMAP",
    )
    ap.add_argument(
        "--umap-n-components",
        type=int,
        default=15,
        help="n_components for dimensionality reduction (clustering space)",
    )
    ap.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="Perplexity for t-SNE (only used when --dr-method tsne)",
    )
    
    # Similarity-specific parameters
    ap.add_argument(
        "--construct-source",
        type=str,
        default="oxford",
        choices=["oxford", "other"],
        help="Construct source from construct_definition.yaml (similarity strategy)",
    )
    ap.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help="Minimum similarity threshold for assignment (similarity strategy)",
    )
    
    # General parameters
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory",
    )
    ap.add_argument(
        "--min-stability",
        type=float,
        default=0.2,
        help="Minimum stability score for consensus clustering (0.0-1.0). "
             "Items with lower scores have unreliable cluster membership.",
    )
    
    return ap.parse_args()


def parse_grid_params(param_str: str) -> list:
    """Parse comma-separated parameter string."""
    parts = [p.strip() for p in param_str.split(",")]
    result = []
    for p in parts:
        if p.lower() == "none":
            result.append(None)
        else:
            try:
                result.append(int(p))
            except ValueError:
                result.append(float(p))
    return result


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Set up paths
    root = Path(__file__).resolve().parents[2]
    model_slug = args.model.replace("/", "_").replace("-", "_")
    processed_dir = root / "data" / "processed"
        
    # Output directory
    if args.output_dir is None:
        output_dir = root / "data" / "outputs" / "experiments" / f"{model_slug}_analysis"
    else:
        output_dir = Path(args.output_dir)

    clust_dir = ensure_dir(str(output_dir / "03_clustering"))

    print(f"\n{'='*70}")
    print(f"STEP 3: CLUSTERING - {args.strategy.upper()} STRATEGY")
    print(f"{'='*70}\n")
    print(f"Model: {args.model}")
    print(f"Strategy: {args.strategy}")
    print(f"Output directory: {clust_dir}\n")

    # Load data
    print("[1/4] Loading embeddings...")
    # Try to load from output directory first (run-specific), fall back to processed/
    items_path = output_dir / "items.csv"
    if not items_path.exists():
        print("      items.csv not found in output directory, trying PROCESSED directory...")
        items_path = processed_dir / "items.csv"
    items = pd.read_csv(items_path)
    
    emb_path = output_dir / f"embeddings_{model_slug}.npy"
    if not emb_path.exists():
        emb_path = processed_dir / f"embeddings_{model_slug}.npy"
    emb = np.load(emb_path)
    n_items = len(items)
    if emb.shape[0] != n_items:
        raise ValueError(
            "Mismatch between items and embeddings. "
            f"items.csv has {n_items} rows, but embeddings has {emb.shape[0]} rows. "
            "Ensure items.csv and embeddings were generated from the same run. "
            "Tip: use --output-dir to point to the matching experiment folder, "
            "or re-run the embedding step to regenerate embeddings for these items."
        )
    print(f"      Loaded {n_items} items")

    # Prepare DR kwargs (common to all strategies)
    umap_kwargs = {
        "metric": "cosine",
        "n_neighbors": args.umap_n_neighbors,
        "min_dist": args.umap_min_dist,
        "n_components": args.umap_n_components,
        "perplexity": args.tsne_perplexity,  # For t-SNE
    }

    # Initialize strategy
    print(f"\n[2/4] Initializing {args.strategy} strategy...")
    
    if args.strategy == "kmeans_consensus":
        strategy = KMeansConsensusStrategy(
            embeddings=emb,
            items=items,
            output_dir=clust_dir,
            model_name=args.model,
            timestamp=timestamp,
            n_clusters=args.n_clusters,
            n_iterations=args.n_iterations,
            umap_kwargs=umap_kwargs,
            consensus_linkage=args.consensus_linkage,
            min_stability=args.min_stability,
            use_dr=not args.no_dr,
            dr_method=args.dr_method,
        )
        reduction_str = "no DR" if args.no_dr else f"with {args.dr_method.upper()} reduction"
        print(f"      KMeans consensus {reduction_str} - {args.n_iterations} iterations")
        print(f"      Number of clusters: {args.n_clusters}")
        print(f"      Consensus linkage: {args.consensus_linkage}")

    elif args.strategy == "kmeans_simple":
        strategy = KMeansSimpleStrategy(
            embeddings=emb,
            items=items,
            output_dir=clust_dir,
            model_name=args.model,
            timestamp=timestamp,
            n_clusters=args.n_clusters,
            n_iterations=args.n_iterations,
            umap_kwargs=umap_kwargs,
            use_dr=not args.no_dr,
            dr_method=args.dr_method,
        )
        reduction_str = "no DR" if args.no_dr else f"with {args.dr_method.upper()} reduction"
        print(f"      KMeans simple {reduction_str} - n_init={args.n_iterations}")
        print(f"      Number of clusters: {args.n_clusters}")
        
    elif args.strategy == "hdbscan":
        # Parse HDBSCAN grid parameters
        hdbscan_grid = {
            "min_cluster_size": parse_grid_params(args.min_cluster_size),
            "min_samples": parse_grid_params(args.min_samples),
        }
        strategy = HDBSCANStrategy(
            embeddings=emb,
            items=items,
            output_dir=clust_dir,
            model_name=args.model,
            timestamp=timestamp,
            umap_kwargs=umap_kwargs,
            hdbscan_param_grid=hdbscan_grid,
            dr_method=args.dr_method,
        )
        print(f"      HDBSCAN with grid search ({args.dr_method.upper()} reduction)")
        print(f"      Min cluster sizes: {hdbscan_grid['min_cluster_size']}")
        print(f"      Min samples: {hdbscan_grid['min_samples']}")
        
    elif args.strategy == "similarity":
        strategy = SimilarityClusteringStrategy(
            embeddings=emb,
            items=items,
            output_dir=clust_dir,
            model_name=args.model,
            timestamp=timestamp,
            construct_source=args.construct_source,
            min_similarity=args.min_similarity,
        )
        print(f"      Similarity-based clustering using construct definitions")
        print(f"      Construct source: {args.construct_source}")
        print(f"      Minimum similarity threshold: {args.min_similarity}")
        
    elif args.strategy == "kmeans_multi_init":
        print("      Strategy not yet implemented")
        sys.exit(1)
    
    else:
        print(f"      Unknown strategy: {args.strategy}")
        sys.exit(1)

    # Run clustering
    print(f"\n[3/4] Running clustering...")
    result = strategy.run()
    
    print(f"\n      Clustering complete!")
    print(f"      Number of clusters: {len(np.unique(result.labels[result.labels >= 0]))}")
    print(f"      Noise points: {np.sum(result.labels == -1)}")

    # Save outputs
    print(f"\n[4/4] Saving outputs...")
    output_files = strategy.save_outputs(save_to_model_specs=(args.strategy == "kmeans_consensus"))
    
    print(f"\n      Saved {len(output_files)} output files:")
    for key, path in output_files.items():
        print(f"        - {key}: {path.name}")

    # Print summary
    print(f"\n{'='*70}")
    print("CLUSTERING COMPLETE!")
    print(f"{'='*70}\n")
    print(f"Strategy: {args.strategy}")
    print(f"Output directory: {clust_dir}")
    print(f"\nKey outputs:")
    print(f"  - Cluster assignments: {clust_dir / 'cluster_assignments.csv'}")
    dr_method_upper = args.dr_method.upper()
    print(f"  - {dr_method_upper} 2D plot: {clust_dir / f'{args.dr_method}_2d_clusters.png'}")
    
    if args.strategy == "kmeans_consensus":
        print(f"\nConsensus Model:")
        print(f"  - Number of factors: {args.n_clusters}")
        print(f"  - Mean stability: {result.probabilities.mean():.3f}")
        print(f"  - Items with low stability (<{args.min_stability}): {np.sum(result.probabilities < args.min_stability)}")
        if 'validation_metrics' in result.diagnostics:
            vm = result.diagnostics['validation_metrics']
            print(f"  - Mean ARI to consensus: {vm['mean_ari_to_consensus']:.4f}")
            print(f"  - Separation ratio: {vm['separation_ratio']:.2f}x")
    
    elif args.strategy == "kmeans_simple":
        km = result.metadata.get("kmeans_metrics", {})
        print(f"\nKMeans Simple Results:")
        print(f"  - Inertia: {km.get('mean_inertia', float('nan')):.2f}")
        if not np.isnan(km.get('mean_silhouette', float('nan'))):
            print(f"  - Silhouette: {km.get('mean_silhouette', float('nan')):.4f}")

    elif args.strategy == "hdbscan":
        print(f"\nHDBSCAN Results:")
        print(f"  - Best params: min_cluster_size={result.params['min_cluster_size']}, "
              f"min_samples={result.params['min_samples']}")
        if 'dbcv' in result.diagnostics:
            print(f"  - DBCV score: {result.diagnostics['dbcv']:.4f}")
    
    elif args.strategy == "similarity":
        print(f"\nSimilarity-Based Results:")
        print(f"  - Construct source: {result.params['construct_source']}")
        print(f"  - Number of factors: {result.params['n_factors']}")
        print(f"  - Factor names: {', '.join(result.diagnostics['factor_names'])}")
        print(f"  - Mean similarity: {result.metadata['similarity_metrics']['mean_similarity']:.4f}")
        print(f"  - Items assigned: {result.metadata['n_assigned']}")
        print(f"  - Items below threshold: {result.metadata['n_noise']}")
    
    print(f"\nNext steps:")
    print(f"  - Run labeling: python bayley_nlp/cli/04_labeling.py --model {args.model}")
    print(f"  - Review outputs in: {clust_dir}\n")


if __name__ == "__main__":
    main()
