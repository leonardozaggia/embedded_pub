
# --- UTF-8 stdout/stderr: this script prints Unicode (arrows, checkmarks)
# which crashes under Windows' default cp1252 encoding; force UTF-8. ---
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")
del _stream

#!/usr/bin/env python
"""
Step 07: Compare B4 and B3 clustering results using mapped item pairs.

This script:
1) Loads the B4-B3 item pairs from Step 06
2) Loads B4 clustering results
3) Loads B3 clustering results (from independent B3 clustering run)
4) Evaluates how well paired items cluster together across versions
5) Generates metrics and visualizations

Metrics:
- Pair agreement rate (% of pairs in same relative cluster position)
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Cluster-to-cluster mapping quality

Visualizations:
- Confusion matrix heatmap
- Sankey diagram (cluster flow B4→B3)
- Agreement vs similarity scatter plot
- Co-occurrence matrix

Usage:
    python 07_compare_mapped_clusters.py --model all-mpnet-base-v2
    python 07_compare_mapped_clusters.py --model all-roberta-large-v1 --b3-clustering path/to/b3/clustering
"""

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

from bayley_nlp.core import ensure_dir


def parse_args():
    ap = argparse.ArgumentParser(
        description="Compare B4 and B3 clustering results using mapped pairs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--model",
        type=str,
        default="all-mpnet-base-v2",
        choices=["all-mpnet-base-v2", "all-roberta-large-v1"],
        help="Sentence transformer model name",
    )
    ap.add_argument(
        "--pairing-dir",
        type=str,
        default=None,
        help="Directory containing pairing results from Step 06 (default: auto-detect)",
    )
    ap.add_argument(
        "--b4-clustering",
        type=str,
        default=None,
        help="Path to B4 cluster assignments CSV (default: auto-detect)",
    )
    ap.add_argument(
        "--b3-clustering",
        type=str,
        default=None,
        help="Path to B3 cluster assignments CSV (default: auto-detect)",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/outputs/experiments/<model>/07_cluster_comparison)",
    )
    ap.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help="Minimum pairing similarity threshold for analysis",
    )
    return ap.parse_args()


def compute_cluster_mapping(pairs_df, b4_clusters, b3_clusters):
    """
    Compute optimal cluster mapping from B4 to B3 using Hungarian algorithm.
    
    Returns:
        mapping: dict {b4_cluster_id: b3_cluster_id}
        mapping_quality: dict with quality metrics per cluster
    """
    # Create contingency matrix (B4 clusters × B3 clusters)
    b4_labels = pairs_df['b4_cluster'].values
    b3_labels = pairs_df['b3_cluster'].values
    
    n_b4_clusters = b4_clusters['cluster'].nunique()
    n_b3_clusters = b3_clusters['cluster'].nunique()
    
    contingency = np.zeros((n_b4_clusters, n_b3_clusters))
    
    for b4_c, b3_c in zip(b4_labels, b3_labels):
        if b4_c >= 0 and b3_c >= 0:  # Exclude noise points
            contingency[int(b4_c), int(b3_c)] += 1
    
    # Use Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(-contingency)
    
    mapping = {int(b4_c): int(b3_c) for b4_c, b3_c in zip(row_ind, col_ind)}
    
    # Calculate mapping quality
    mapping_quality = {}
    for b4_c, b3_c in mapping.items():
        total_b4_items = contingency[b4_c, :].sum()
        matched_items = contingency[b4_c, b3_c]
        quality = matched_items / total_b4_items if total_b4_items > 0 else 0.0
        mapping_quality[b4_c] = {
            'mapped_to': b3_c,
            'agreement_rate': quality,
            'n_items': int(total_b4_items),
            'n_matched': int(matched_items),
        }
    
    return mapping, mapping_quality, contingency


def calculate_metrics(pairs_df, optimal_mapping):
    """
    Calculate clustering agreement metrics.
    
    Returns:
        metrics: dict with various evaluation metrics
    """
    # Filter valid pairs (exclude noise)
    valid_pairs = pairs_df[
        (pairs_df['b4_cluster'] >= 0) & (pairs_df['b3_cluster'] >= 0)
    ].copy()
    
    if len(valid_pairs) == 0:
        return {
            'n_total_pairs': len(pairs_df),
            'n_valid_pairs': 0,
            'error': 'No valid pairs (all noise)',
        }
    
    b4_labels = valid_pairs['b4_cluster'].values
    b3_labels = valid_pairs['b3_cluster'].values
    
    # Map B4 labels to B3 space using optimal mapping
    b4_labels_mapped = np.array([optimal_mapping.get(c, -1) for c in b4_labels])
    
    # Calculate agreement rate
    agreement = (b4_labels_mapped == b3_labels).sum() / len(b4_labels)
    
    # Calculate ARI and NMI
    ari = adjusted_rand_score(b4_labels, b3_labels)
    nmi = normalized_mutual_info_score(b4_labels, b3_labels)
    
    # Calculate per-cluster agreement
    cluster_agreement = {}
    for b4_c in valid_pairs['b4_cluster'].unique():
        mask = valid_pairs['b4_cluster'] == b4_c
        cluster_pairs = valid_pairs[mask]
        mapped_c = optimal_mapping.get(b4_c, -1)
        if mapped_c >= 0:
            cluster_match = (cluster_pairs['b3_cluster'] == mapped_c).sum()
            cluster_agreement[int(b4_c)] = cluster_match / len(cluster_pairs)
        else:
            cluster_agreement[int(b4_c)] = 0.0
    
    # Similarity statistics
    similarity_stats = {
        'mean': float(valid_pairs['similarity'].mean()),
        'median': float(valid_pairs['similarity'].median()),
        'std': float(valid_pairs['similarity'].std()),
        'min': float(valid_pairs['similarity'].min()),
        'max': float(valid_pairs['similarity'].max()),
    }
    
    metrics = {
        'n_total_pairs': len(pairs_df),
        'n_valid_pairs': len(valid_pairs),
        'agreement_rate': float(agreement),
        'adjusted_rand_index': float(ari),
        'normalized_mutual_information': float(nmi),
        'cluster_agreement': cluster_agreement,
        'similarity_stats': similarity_stats,
    }
    
    return metrics


def plot_confusion_matrix(contingency, output_dir):
    """Plot confusion matrix heatmap of cluster assignments."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        contingency,
        annot=True,
        fmt='.0f',
        cmap='Blues',
        xticklabels=[f'B3-C{i}' for i in range(contingency.shape[1])],
        yticklabels=[f'B4-C{i}' for i in range(contingency.shape[0])],
        ax=ax,
        cbar_kws={'label': 'Number of Paired Items'}
    )
    
    ax.set_xlabel('B3 Clusters', fontsize=12)
    ax.set_ylabel('B4 Clusters', fontsize=12)
    ax.set_title('B4-B3 Cluster Co-occurrence Matrix', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_agreement_vs_similarity(pairs_df, optimal_mapping, output_dir):
    """Plot agreement rate vs pairing similarity."""
    # Add agreement column
    pairs_df = pairs_df.copy()
    pairs_df['b4_mapped'] = pairs_df['b4_cluster'].map(optimal_mapping)
    pairs_df['agrees'] = (pairs_df['b4_mapped'] == pairs_df['b3_cluster']) & \
                         (pairs_df['b4_cluster'] >= 0) & \
                         (pairs_df['b3_cluster'] >= 0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Scatter plot
    ax = axes[0]
    valid = pairs_df[(pairs_df['b4_cluster'] >= 0) & (pairs_df['b3_cluster'] >= 0)]
    
    agree = valid[valid['agrees']]
    disagree = valid[~valid['agrees']]
    
    ax.scatter(disagree['similarity'], np.random.random(len(disagree)) * 0.4 - 0.2,
               alpha=0.5, s=50, c='red', label=f'Disagree (n={len(disagree)})')
    ax.scatter(agree['similarity'], np.random.random(len(agree)) * 0.4 + 0.8,
               alpha=0.5, s=50, c='green', label=f'Agree (n={len(agree)})')
    
    ax.set_xlabel('Pairing Similarity (Cosine)', fontsize=11)
    ax.set_ylabel('Cluster Agreement', fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Disagree', 'Agree'])
    ax.set_title('Cluster Agreement vs Pairing Similarity', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Right: Binned agreement rate
    ax = axes[1]
    bins = np.linspace(valid['similarity'].min(), valid['similarity'].max(), 10)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    agreement_rates = []
    bin_counts = []
    
    for i in range(len(bins) - 1):
        mask = (valid['similarity'] >= bins[i]) & (valid['similarity'] < bins[i+1])
        if mask.sum() > 0:
            agreement_rates.append(valid[mask]['agrees'].mean())
            bin_counts.append(mask.sum())
        else:
            agreement_rates.append(0)
            bin_counts.append(0)
    
    ax.bar(bin_centers, agreement_rates, width=(bins[1]-bins[0])*0.8, 
           alpha=0.7, edgecolor='black')
    
    # Add count labels
    for x, y, count in zip(bin_centers, agreement_rates, bin_counts):
        if count > 0:
            ax.text(x, y + 0.02, f'n={count}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Pairing Similarity (Cosine)', fontsize=11)
    ax.set_ylabel('Agreement Rate', fontsize=11)
    ax.set_title('Agreement Rate by Similarity Bin', fontsize=12)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'agreement_vs_similarity.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_sankey_diagram(pairs_df, optimal_mapping, output_dir):
    """
    Create a simplified Sankey-style visualization showing cluster flow from B4 to B3.
    Uses matplotlib for a basic alluvial-style plot.
    """
    # Filter valid pairs
    valid = pairs_df[(pairs_df['b4_cluster'] >= 0) & (pairs_df['b3_cluster'] >= 0)].copy()
    
    if len(valid) == 0:
        print("      Warning: No valid pairs for Sankey diagram")
        return
    
    # Count flows between clusters
    flows = valid.groupby(['b4_cluster', 'b3_cluster']).size().reset_index(name='count')
    
    # Create flow diagram
    fig, ax = plt.subplots(figsize=(12, 8))
    
    n_b4_clusters = int(valid['b4_cluster'].max() + 1)
    n_b3_clusters = int(valid['b3_cluster'].max() + 1)
    
    # Position clusters vertically
    b4_y = np.linspace(0, 1, n_b4_clusters)
    b3_y = np.linspace(0, 1, n_b3_clusters)
    
    # Draw connections
    cmap = plt.cm.tab10
    for _, row in flows.iterrows():
        b4_c = int(row['b4_cluster'])
        b3_c = int(row['b3_cluster'])
        count = row['count']
        
        # Determine if this is the optimal mapping
        is_optimal = optimal_mapping.get(b4_c) == b3_c
        alpha = 0.7 if is_optimal else 0.2
        color = cmap(b4_c % 10)
        
        # Draw connecting line
        ax.plot([0, 1], [b4_y[b4_c], b3_y[b3_c]], 
                linewidth=count*2, alpha=alpha, color=color, solid_capstyle='round')
    
    # Draw cluster nodes
    for i in range(n_b4_clusters):
        n_items = (valid['b4_cluster'] == i).sum()
        ax.scatter([0], [b4_y[i]], s=n_items*20, c=[cmap(i % 10)], 
                  edgecolors='black', linewidths=2, zorder=10)
        ax.text(-0.05, b4_y[i], f'B4-C{i}\n(n={n_items})', 
               ha='right', va='center', fontsize=10, weight='bold')
    
    for i in range(n_b3_clusters):
        n_items = (valid['b3_cluster'] == i).sum()
        ax.scatter([1], [b3_y[i]], s=n_items*20, c=[cmap(i % 10)], 
                  edgecolors='black', linewidths=2, zorder=10)
        ax.text(1.05, b3_y[i], f'B3-C{i}\n(n={n_items})', 
               ha='left', va='center', fontsize=10, weight='bold')
    
    ax.set_xlim([-0.2, 1.2])
    ax.set_ylim([-0.1, 1.1])
    ax.axis('off')
    ax.set_title('Cluster Flow: B4 → B3 (thick lines = optimal mapping)', 
                fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sankey_cluster_flow.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_cluster_agreement_bars(mapping_quality, output_dir):
    """Plot per-cluster agreement rates."""
    clusters = sorted(mapping_quality.keys())
    agreement_rates = [mapping_quality[c]['agreement_rate'] for c in clusters]
    n_items = [mapping_quality[c]['n_items'] for c in clusters]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(len(clusters)), agreement_rates, alpha=0.7, edgecolor='black')
    
    # Color bars by agreement rate
    for bar, rate in zip(bars, agreement_rates):
        if rate >= 0.8:
            bar.set_color('green')
        elif rate >= 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Add item count labels
    for i, (rate, n) in enumerate(zip(agreement_rates, n_items)):
        ax.text(i, rate + 0.02, f'n={n}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('B4 Cluster', fontsize=11)
    ax.set_ylabel('Agreement Rate with Mapped B3 Cluster', fontsize=11)
    ax.set_title('Per-Cluster Agreement Rates', fontsize=12)
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels([f'C{c}' for c in clusters])
    ax.set_ylim([0, 1.1])
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='High agreement (≥80%)')
    ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Medium agreement (≥60%)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_agreement_bars.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_detailed_results(pairs_df, optimal_mapping, mapping_quality, metrics, output_dir):
    """Save detailed results to CSV files."""
    # 1. Pairs with agreement status
    pairs_detailed = pairs_df.copy()
    pairs_detailed['b4_mapped_to'] = pairs_detailed['b4_cluster'].map(optimal_mapping)
    pairs_detailed['clusters_agree'] = (
        (pairs_detailed['b4_mapped_to'] == pairs_detailed['b3_cluster']) &
        (pairs_detailed['b4_cluster'] >= 0) &
        (pairs_detailed['b3_cluster'] >= 0)
    )
    pairs_detailed.to_csv(output_dir / 'paired_items_with_agreement.csv', index=False)
    
    # 2. Cluster mapping table
    mapping_df = pd.DataFrame([
        {
            'b4_cluster': b4_c,
            'b3_cluster_mapped': info['mapped_to'],
            'agreement_rate': info['agreement_rate'],
            'n_items': info['n_items'],
            'n_matched': info['n_matched'],
        }
        for b4_c, info in mapping_quality.items()
    ])
    mapping_df = mapping_df.sort_values('b4_cluster')
    mapping_df.to_csv(output_dir / 'cluster_mapping.csv', index=False)
    
    # 3. Summary metrics
    with open(output_dir / 'comparison_metrics.yaml', 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False, sort_keys=False)


def main():
    args = parse_args()

    root = Path(__file__).resolve().parents[2]
    model_slug = args.model.replace("/", "_").replace("-", "_")

    # Auto-detect paths if not provided
    experiments_dir = root / "data" / "outputs" / "experiments" / f"{model_slug}_analysis"
    
    if args.pairing_dir is None:
        pairing_dir = experiments_dir / "06_b4_b3_pairing"
    else:
        pairing_dir = Path(args.pairing_dir)
        if not (pairing_dir / "b4_b3_greedy_pairs.csv").exists():
            fallback_pairing_dir = pairing_dir / "06_b4_b3_pairing"
            if (fallback_pairing_dir / "b4_b3_greedy_pairs.csv").exists():
                pairing_dir = fallback_pairing_dir

    if args.b4_clustering is None:
        b4_clustering_file = experiments_dir / "03_clustering" / "cluster_assignments.csv"
    else:
        b4_clustering_file = Path(args.b4_clustering)

    if args.b3_clustering is None:
        # Try to find B3 clustering in a separate B3 analysis directory
        b3_experiments_dir = root / "data" / "outputs" / "experiments" / f"{model_slug}_b3_analysis"
        b3_clustering_file = b3_experiments_dir / "03_clustering" / "cluster_assignments.csv"
        if not b3_clustering_file.exists():
            # Fallback: look for B3 clustering in the B3 pairing output (created by step 06)
            b3_clustering_file = pairing_dir / "b3_cluster_assignments.csv"
    else:
        b3_clustering_file = Path(args.b3_clustering)

    if args.output_dir is None:
        output_dir = experiments_dir / "07_cluster_comparison"
    else:
        output_dir = Path(args.output_dir)
        if output_dir.name != "07_cluster_comparison":
            output_dir = output_dir / "07_cluster_comparison"

    ensure_dir(str(output_dir))

    print(f"\n{'='*70}")
    print("STEP 07: B4-B3 CLUSTER COMPARISON")
    print(f"{'='*70}\n")
    print(f"Model: {args.model}")
    print(f"Pairing directory: {pairing_dir}")
    print(f"B4 clustering: {b4_clustering_file}")
    print(f"B3 clustering: {b3_clustering_file}")
    print(f"Output directory: {output_dir}\n")

    # Verify files exist
    pairs_file = pairing_dir / "b4_b3_greedy_pairs.csv"
    if not pairs_file.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_file}\nRun step 06 first: python bayley_nlp/cli/06_b4_to_b3.py")
    
    if not b4_clustering_file.exists():
        raise FileNotFoundError(f"B4 clustering file not found: {b4_clustering_file}\nRun B4 clustering first")
    
    if not b3_clustering_file.exists():
        raise FileNotFoundError(f"B3 clustering file not found: {b3_clustering_file}\nRun B3 clustering or step 06 first")

    print("[1/6] Loading data...")
    pairs_df = pd.read_csv(pairs_file)
    b4_clusters = pd.read_csv(b4_clustering_file)
    b3_clusters = pd.read_csv(b3_clustering_file)
    
    print(f"      Loaded {len(pairs_df)} item pairs")
    print(f"      B4: {len(b4_clusters)} items, {b4_clusters['cluster'].nunique()} clusters")
    print(f"      B3: {len(b3_clusters)} items, {b3_clusters['cluster'].nunique()} clusters")

    # Filter by similarity threshold
    if args.min_similarity > 0:
        n_before = len(pairs_df)
        pairs_df = pairs_df[pairs_df['similarity'] >= args.min_similarity].copy()
        print(f"      Filtered to {len(pairs_df)} pairs (similarity ≥ {args.min_similarity})")

    print("\n[2/6] Merging cluster assignments with pairs...")
    # Add cluster assignments to pairs
    pairs_df = pairs_df.merge(
        b4_clusters[['item_id', 'cluster']].rename(columns={'cluster': 'b4_cluster'}),
        left_on='b4_item_id',
        right_on='item_id',
        how='left'
    ).drop(columns=['item_id'])
    
    pairs_df = pairs_df.merge(
        b3_clusters[['item_id', 'cluster']].rename(columns={'cluster': 'b3_cluster'}),
        left_on='b3_item_id',
        right_on='item_id',
        how='left'
    ).drop(columns=['item_id'])
    
    print(f"      Merged cluster assignments for {len(pairs_df)} pairs")

    print("\n[3/6] Computing optimal cluster mapping...")
    optimal_mapping, mapping_quality, contingency = compute_cluster_mapping(
        pairs_df, b4_clusters, b3_clusters
    )
    
    print(f"      Optimal cluster mapping (B4 → B3):")
    for b4_c, info in sorted(mapping_quality.items()):
        b3_c = info['mapped_to']
        rate = info['agreement_rate']
        n = info['n_items']
        print(f"        C{b4_c} → C{b3_c} (agreement: {rate:.2%}, n={n})")

    print("\n[4/6] Calculating evaluation metrics...")
    metrics = calculate_metrics(pairs_df, optimal_mapping)
    
    print(f"\n      RESULTS:")
    print(f"      --------")
    print(f"      Total pairs: {metrics['n_total_pairs']}")
    print(f"      Valid pairs: {metrics['n_valid_pairs']}")
    print(f"      Agreement rate: {metrics['agreement_rate']:.2%}")
    print(f"      Adjusted Rand Index: {metrics['adjusted_rand_index']:.4f}")
    print(f"      Normalized Mutual Information: {metrics['normalized_mutual_information']:.4f}")
    print(f"\n      Pairing similarity: μ={metrics['similarity_stats']['mean']:.4f}, "
          f"σ={metrics['similarity_stats']['std']:.4f}")

    print("\n[5/6] Creating visualizations...")
    plot_confusion_matrix(contingency, output_dir)
    print("      ✓ Confusion matrix")
    
    plot_agreement_vs_similarity(pairs_df, optimal_mapping, output_dir)
    print("      ✓ Agreement vs similarity plots")
    
    plot_sankey_diagram(pairs_df, optimal_mapping, output_dir)
    print("      ✓ Sankey cluster flow diagram")
    
    plot_cluster_agreement_bars(mapping_quality, output_dir)
    print("      ✓ Per-cluster agreement bars")

    print("\n[6/6] Saving detailed results...")
    save_detailed_results(pairs_df, optimal_mapping, mapping_quality, metrics, output_dir)
    print(f"      ✓ Saved paired_items_with_agreement.csv")
    print(f"      ✓ Saved cluster_mapping.csv")
    print(f"      ✓ Saved comparison_metrics.yaml")

    print(f"\n{'='*70}")
    print("STEP 07 COMPLETE!")
    print(f"{'='*70}\n")
    print(f"Summary:")
    print(f"  Agreement rate: {metrics['agreement_rate']:.2%}")
    print(f"  ARI: {metrics['adjusted_rand_index']:.4f}")
    print(f"  NMI: {metrics['normalized_mutual_information']:.4f}")
    print(f"\nAll results saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
