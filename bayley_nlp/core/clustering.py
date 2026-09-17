"""
Clustering strategies for semantic PCA analysis.

This module provides multiple clustering approaches for discovering item factors:
- Pipeline A: KMeans with consensus clustering (1000 runs)
- Pipeline B: Similarity-based clustering (item-factor assignments)
- Pipeline C: KMeans with multi-init (n_init=1000)
- Pipeline D: HDBSCAN density-based clustering (legacy)

Each strategy implements a common interface for use in CLI scripts.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from tqdm import tqdm

from .umap_hdbscan import umap_reduce, umap_hdbscan_auto
from .dimensionality_reduction import get_reducer, CORE_METHODS
from .utils import ensure_dir


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ClusteringResult:
    """Results from a clustering analysis."""
    labels: np.ndarray
    probabilities: np.ndarray  # Can be stability scores for consensus methods
    params: Dict
    diagnostics: Dict
    metadata: Dict


# ============================================================================
# Utility Functions
# ============================================================================

def validate_n_clusters(n_clusters: int, n_items: int) -> bool:
    """Validate that n_clusters is reasonable for the dataset."""
    if n_clusters < 2:
        raise ValueError(f"n_clusters must be at least 2, got {n_clusters}")
    if n_clusters >= n_items:
        raise ValueError(f"n_clusters ({n_clusters}) must be less than n_items ({n_items})")
    if n_clusters > n_items // 2:
        print(f"Warning: n_clusters ({n_clusters}) is more than half of n_items ({n_items})")
    return True


def calculate_cluster_stability(all_labels: List[np.ndarray], n_items: int) -> Tuple[np.ndarray, List[float]]:
    """
    Calculate stability metrics for clustering across multiple runs.
    
    Returns:
        - co_occurrence_matrix: n_items x n_items matrix of co-clustering frequency
        - pairwise_ari: pairwise Adjusted Rand Index between runs (sampled if > 200 runs)
    """
    n_runs = len(all_labels)

    # Build co-occurrence matrix via vectorized numpy (avoids O(n_runs*n_items^2) Python loops)
    all_labels_arr = np.array(all_labels)  # (n_runs, n_items)
    # same_cluster[r, i, j] == 1 if items i and j are in the same cluster on run r
    co_occurrence = np.sum(
        all_labels_arr[:, :, None] == all_labels_arr[:, None, :], axis=0
    ).astype(float) / n_runs

    # Calculate pairwise ARI — sample pairs when there are many runs to stay fast
    max_pairs = 5000
    all_pairs = [(i, j) for i in range(n_runs) for j in range(i + 1, n_runs)]
    if len(all_pairs) > max_pairs:
        rng = np.random.default_rng(42)
        sampled = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        pairs = [all_pairs[k] for k in sampled]
    else:
        pairs = all_pairs

    ari_scores = [
        adjusted_rand_score(all_labels[i], all_labels[j]) for i, j in pairs
    ]

    return co_occurrence, ari_scores


def analyze_factor_composition_stability(all_labels: List[np.ndarray], 
                                        item_ids: np.ndarray, 
                                        output_dir: Path) -> np.ndarray:
    """
    Analyze how stable the composition of each factor is across runs.
    For each item, calculate the percentage of times it appears in each cluster position.
    """
    n_runs = len(all_labels)
    n_items = len(item_ids)
    
    # Track which items appear together most frequently
    item_cluster_assignments = defaultdict(Counter)
    
    for run_idx, labels in enumerate(all_labels):
        # Map cluster IDs to sorted order (by size or ID)
        unique_clusters = sorted(np.unique(labels))
        
        for item_idx, cluster_id in enumerate(labels):
            # Use the position in sorted clusters as the "factor" identifier
            factor_pos = unique_clusters.index(cluster_id)
            item_cluster_assignments[item_idx][factor_pos] += 1
    
    # Create stability matrix: items x factor positions
    max_factors = max(max(counter.keys()) + 1 for counter in item_cluster_assignments.values() if counter)
    stability_matrix = np.zeros((n_items, max_factors))
    
    for item_idx in range(n_items):
        for factor_pos, count in item_cluster_assignments[item_idx].items():
            stability_matrix[item_idx, factor_pos] = count / n_runs
    
    return stability_matrix


def validate_consensus_clustering(consensus_labels: np.ndarray, 
                                  all_labels: List[np.ndarray], 
                                  co_occurrence: np.ndarray) -> Tuple[Dict, List[float]]:
    """
    Validate that the consensus clustering is representative of individual runs.
    
    Compares the consensus to each individual run and checks alignment with
    the co-occurrence patterns.
    
    Parameters:
    -----------
    consensus_labels : np.ndarray
        Consensus cluster labels
    all_labels : list of np.ndarray
        All individual clustering results
    co_occurrence : np.ndarray
        Co-occurrence matrix
        
    Returns:
    --------
    validation_metrics : dict
        Dictionary with validation statistics
    ari_to_consensus : list of float
        ARI scores between each run and consensus
    """
    n_runs = len(all_labels)
    n_items = len(consensus_labels)
    
    # 1. Calculate ARI between consensus and each individual run
    ari_to_consensus = []
    for labels in all_labels:
        ari = adjusted_rand_score(consensus_labels, labels)
        ari_to_consensus.append(ari)
    
    # 2. For each item, check if consensus cluster matches most frequent cluster
    most_frequent_matches = 0
    for i in range(n_items):
        # Find most common cluster assignment for this item
        item_clusters = [labels[i] for labels in all_labels]
        most_common = Counter(item_clusters).most_common(1)[0][0]
        # Note: cluster IDs may differ, so we can't directly compare
        # Instead, check co-occurrence with consensus cluster members
        consensus_cluster_members = np.where(consensus_labels == consensus_labels[i])[0]
        consensus_cluster_members = consensus_cluster_members[consensus_cluster_members != i]
        
        if len(consensus_cluster_members) > 0:
            mean_cooccur = co_occurrence[i, consensus_cluster_members].mean()
            # If mean co-occurrence is high, consensus is representative
            if mean_cooccur > 0.5:  # Threshold for "good" alignment
                most_frequent_matches += 1
    
    # 3. Calculate how well consensus preserves co-occurrence structure
    # Items in same consensus cluster should have high co-occurrence
    within_cluster_cooccur = []
    between_cluster_cooccur = []
    
    for i in range(n_items):
        for j in range(i + 1, n_items):
            if consensus_labels[i] == consensus_labels[j]:
                within_cluster_cooccur.append(co_occurrence[i, j])
            else:
                between_cluster_cooccur.append(co_occurrence[i, j])
    
    validation_metrics = {
        "mean_ari_to_consensus": float(np.mean(ari_to_consensus)),
        "median_ari_to_consensus": float(np.median(ari_to_consensus)),
        "std_ari_to_consensus": float(np.std(ari_to_consensus)),
        "min_ari_to_consensus": float(np.min(ari_to_consensus)),
        "max_ari_to_consensus": float(np.max(ari_to_consensus)),
        "mean_within_cluster_cooccurrence": float(np.mean(within_cluster_cooccur)) if within_cluster_cooccur else 0.0,
        "mean_between_cluster_cooccurrence": float(np.mean(between_cluster_cooccur)) if between_cluster_cooccur else 0.0,
        "separation_ratio": float(np.mean(within_cluster_cooccur) / np.mean(between_cluster_cooccur)) if between_cluster_cooccur and np.mean(between_cluster_cooccur) > 0 else float('inf'),
        "items_well_represented_pct": float(100 * most_frequent_matches / n_items),
    }
    
    return validation_metrics, ari_to_consensus


def create_consensus_clustering(co_occurrence: np.ndarray, 
                                n_clusters: int, 
                                linkage: str = 'average') -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a consensus clustering from the co-occurrence matrix.
    
    Uses hierarchical clustering on the co-occurrence matrix (treated as similarity)
    to create a stable clustering solution.
    
    Parameters:
    -----------
    co_occurrence : np.ndarray
        n_items x n_items matrix of co-clustering frequencies
    n_clusters : int
        Number of clusters to create
    linkage : str
        Linkage method for hierarchical clustering ('average', 'complete', 'single')
        
    Returns:
    --------
    consensus_labels : np.ndarray
        Cluster labels for the consensus clustering (all items assigned)
    stability_scores : np.ndarray
        Per-item stability scores (mean co-occurrence with cluster members)
    """
    n_items = co_occurrence.shape[0]
    
    # Convert co-occurrence (similarity) to distance
    # High co-occurrence = low distance
    distance_matrix = 1.0 - co_occurrence
    
    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage=linkage
    )
    consensus_labels = clustering.fit_predict(distance_matrix)
    
    # Calculate stability scores for each item
    stability_scores = np.zeros(n_items)
    for i in range(n_items):
        # Items in the same cluster
        cluster_members = np.where(consensus_labels == consensus_labels[i])[0]
        # Mean co-occurrence with cluster members (excluding self)
        cluster_members = cluster_members[cluster_members != i]
        if len(cluster_members) > 0:
            stability_scores[i] = co_occurrence[i, cluster_members].mean()
        else:
            stability_scores[i] = 1.0  # Singleton cluster
    
    return consensus_labels, stability_scores


def save_consensus_to_model_specs(consensus_labels: np.ndarray, 
                                  stability_scores: np.ndarray, 
                                  items_df: pd.DataFrame,
                                  model_name: str, 
                                  timestamp: str, 
                                  umap_kwargs: Dict, 
                                  n_iterations: int, 
                                  stability_summary: Dict, 
                                  root: Optional[Path] = None) -> Tuple[str, Path]:
    """
    Save the consensus clustering solution to model_specs.yaml.
    
    This creates a stable, reproducible factor structure based on multiple runs.
    """
    if root is None:
        root = Path(__file__).resolve().parents[2]
    config_dir = root / "config"
    model_specs_path = config_dir / "model_specs.yaml"
    
    # Load existing model specs
    if model_specs_path.exists():
        with open(model_specs_path, "r", encoding="utf-8") as f:
            model_specs = yaml.safe_load(f) or {}
    else:
        model_specs = {}
    
    # Group items by cluster
    factor_indices = {}
    factor_items = {}
    factor_sizes = {}
    factor_stability_scores = {}
    
    # Get unique clusters (all items are assigned in KMeans)
    unique_clusters = sorted(np.unique(consensus_labels))
    n_clusters = len(unique_clusters)
    
    for i, cluster_id in enumerate(unique_clusters, start=1):
        factor_key = f"F{i}"
        cluster_mask = consensus_labels == cluster_id
        cluster_items = items_df[cluster_mask]
        cluster_stability = stability_scores[cluster_mask]
        
        # Get item indices (1-based) and item IDs
        item_ids = cluster_items["item_id"].tolist()
        # Extract numeric indices from item_ids (e.g., COG_034 -> 1)
        indices = [int(item_id.split("_")[1]) - 33 for item_id in item_ids]
        
        factor_indices[factor_key] = sorted(indices)
        factor_items[factor_key] = sorted(item_ids)
        factor_sizes[factor_key] = len(item_ids)
        # Average stability score for this factor
        factor_stability_scores[factor_key] = float(cluster_stability.mean())
    
    # Create model spec entry
    model_slug = model_name.replace("/", "_").replace("-", "_")
    spec_name = f"{model_slug}_consensus_stable"
    
    model_specs[spec_name] = {
        "source": "consensus_clustering",
        "model": model_name,
        "date": timestamp.split("_")[0],
        "n_factors": n_clusters,
        "n_items": len(items_df), 
        "method": "Consensus from KMeans (multiple runs)",
        "n_iterations": n_iterations,
        "n_clusters_param": n_clusters,
        "umap_params": umap_kwargs,
        "factor_indices": factor_indices,
        "factor_items": factor_items,
        "factor_sizes": factor_sizes,
        "factor_stability_scores": factor_stability_scores,
        "stability_metrics": {
            "mean_item_stability": float(stability_scores.mean()),
            "median_item_stability": float(np.median(stability_scores)),
            "min_item_stability": float(stability_scores.min()),
            "mean_pairwise_ari": stability_summary["stability_metrics"]["mean_pairwise_ari"],
        },
        "timestamp": timestamp,
        "description": (
            f"Consensus clustering derived from {n_iterations} runs with {n_clusters} clusters. "
            f"This model represents the most stable factor structure, where items "
            f"consistently cluster together across different random initializations. "
            f"Factor stability scores indicate how reliably items co-occur within their "
            f"assigned factors."
        ),
    }
    
    # Save back to YAML
    with open(model_specs_path, "w", encoding="utf-8") as f:
        yaml.dump(model_specs, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    return spec_name, model_specs_path


def plot_stability_results(inertias: List[float], 
                          silhouettes: List[float], 
                          co_occurrence: np.ndarray, 
                          stability_matrix: np.ndarray,
                          item_ids: np.ndarray, 
                          ari_scores: List[float], 
                          ari_to_consensus: Optional[List[float]], 
                          output_dir: Path) -> None:
    """Create comprehensive stability visualizations."""
    
    # 1. Distribution of inertia values (KMeans specific)
    if inertias and len(inertias) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(inertias, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(np.mean(inertias), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(inertias):.2f}')
        ax.set_xlabel('KMeans Inertia', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Distribution of KMeans Inertia Across Runs\n'
                     f'Lower values indicate tighter clusters', 
                     fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'inertia_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Co-occurrence heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(co_occurrence, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('Item Index', fontsize=12)
    ax.set_ylabel('Item Index', fontsize=12)
    ax.set_title('Item Co-Clustering Frequency Across Runs\n'
                 '(1.0 = always clustered together, 0.0 = never clustered together)', 
                 fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Co-clustering Frequency', fontsize=12)
    
    # Add grid for better readability
    ax.set_xticks(np.arange(len(item_ids))[::5])
    ax.set_yticks(np.arange(len(item_ids))[::5])
    ax.set_xticklabels(np.arange(len(item_ids))[::5], fontsize=8)
    ax.set_yticklabels(np.arange(len(item_ids))[::5], fontsize=8)
    ax.grid(which='major', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_stability_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Factor composition stability
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(stability_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('Factor Position (sorted by cluster ID)', fontsize=12)
    ax.set_ylabel('Item', fontsize=12)
    ax.set_title('Item-to-Factor Assignment Stability\n'
                 '(1.0 = always assigned to this factor position)', 
                 fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Assignment Frequency', fontsize=12)
    
    # Set ticks
    ax.set_yticks(np.arange(len(item_ids))[::5])
    ax.set_yticklabels(item_ids[::5], fontsize=8)
    ax.set_xticks(np.arange(stability_matrix.shape[1]))
    ax.set_xticklabels([f'F{i+1}' for i in range(stability_matrix.shape[1])], fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'factor_composition_stability.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. ARI distribution
    if ari_scores and len(ari_scores) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(ari_scores, bins=50, alpha=0.7, color='coral', edgecolor='black')
        ax.axvline(np.mean(ari_scores), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean ARI: {np.mean(ari_scores):.3f}')
        ax.axvline(np.median(ari_scores), color='blue', linestyle='--', linewidth=2,
                   label=f'Median ARI: {np.median(ari_scores):.3f}')
        ax.set_xlabel('Adjusted Rand Index', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Pairwise Agreement Between Clustering Runs (ARI)\n'
                     f'Higher values indicate more consistent clustering', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'ari_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Silhouette score distribution
    if silhouettes and len(silhouettes) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(silhouettes, bins=50, alpha=0.7, color='teal', edgecolor='black')
        ax.axvline(np.mean(silhouettes), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(silhouettes):.4f}')
        ax.axvline(np.median(silhouettes), color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(silhouettes):.4f}')
        ax.set_xlabel('Silhouette Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Silhouette Score Distribution Across Runs\n'
                     'Higher values indicate better-defined clusters', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'silhouette_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. ARI to consensus distribution (validation)
    if ari_to_consensus is not None and len(ari_to_consensus) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(ari_to_consensus, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(np.mean(ari_to_consensus), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(ari_to_consensus):.4f}')
        ax.axvline(np.median(ari_to_consensus), color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(ari_to_consensus):.4f}')
        ax.set_xlabel('Adjusted Rand Index to Consensus', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Agreement Between Individual Runs and Consensus Model\n'
                     'Higher values indicate consensus is representative', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'consensus_validation_ari.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"      All stability plots saved to: {output_dir}")


def plot_consensus_model(consensus_labels: np.ndarray, 
                        stability_scores: np.ndarray, 
                        item_ids: np.ndarray, 
                        output_dir: Path) -> None:
    """Plot the consensus clustering model with stability scores."""
    
    # 1. Consensus cluster assignments with stability scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Cluster assignments
    unique_clusters = sorted(np.unique(consensus_labels))
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    
    for i, item_idx in enumerate(range(len(item_ids))):
        cluster = consensus_labels[item_idx]
        color_idx = unique_clusters.index(cluster)
        ax1.barh(i, 1, color=cluster_colors[color_idx], alpha=0.7)
    
    ax1.set_yticks(range(0, len(item_ids), 5))
    ax1.set_yticklabels(item_ids[::5], fontsize=8)
    ax1.set_xlabel('Cluster Assignment', fontsize=12)
    ax1.set_ylabel('Item', fontsize=12)
    ax1.set_title('Consensus Cluster Assignments', fontsize=14)
    ax1.set_xlim(0, 1)
    ax1.set_xticks([])
    
    # Right: Stability scores
    ax2.barh(range(len(item_ids)), stability_scores, color='steelblue', alpha=0.7)
    ax2.axvline(np.mean(stability_scores), color='red', linestyle='--', linewidth=1, alpha=0.5, 
                label=f'Mean: {np.mean(stability_scores):.3f}')
    ax2.set_yticks(range(0, len(item_ids), 5))
    ax2.set_yticklabels(item_ids[::5], fontsize=8)
    ax2.set_xlabel('Stability Score', fontsize=12)
    ax2.set_ylabel('Item', fontsize=12)
    ax2.set_title('Item Stability Scores\n(Higher = More Stable)', fontsize=14)
    ax2.set_xlim(0, 1)
    ax2.legend(fontsize=10)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'consensus_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Cluster size comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_sizes = [np.sum(consensus_labels == c) for c in unique_clusters]
    ax.bar(range(1, len(unique_clusters) + 1), cluster_sizes, 
           color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Factor Number', fontsize=12)
    ax.set_ylabel('Number of Items', fontsize=12)
    ax.set_title(f'Consensus Model Factor Sizes\n'
                 f'{len(unique_clusters)} Factors (all items assigned)',
                 fontsize=14)
    ax.set_xticks(range(1, len(unique_clusters) + 1))
    ax.set_xticklabels([f'F{i}' for i in range(1, len(unique_clusters) + 1)])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, size in enumerate(cluster_sizes):
        ax.text(i + 1, size + 0.5, str(size), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'consensus_factor_sizes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"      Consensus model plots saved to: {output_dir}")


# ============================================================================
# Base Strategy Class
# ============================================================================

class ClusteringStrategy(ABC):
    """Abstract base class for clustering strategies."""
    
    def __init__(self, embeddings: np.ndarray, items: pd.DataFrame, output_dir: Path, 
                 model_name: str, timestamp: str, **kwargs):
        """
        Initialize clustering strategy.
        
        Parameters:
        -----------
        embeddings : np.ndarray
            Item embeddings (n_items x embedding_dim)
        items : pd.DataFrame
            Items dataframe with at least 'item_id' column
        output_dir : Path
            Output directory for results
        model_name : str
            Name of the embedding model
        timestamp : str
            Timestamp for this run
        **kwargs : dict
            Additional strategy-specific parameters
        """
        self.embeddings = embeddings
        self.items = items
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.timestamp = timestamp
        self.kwargs = kwargs
        self.result: Optional[ClusteringResult] = None
        
    @abstractmethod
    def run(self) -> ClusteringResult:
        """Execute the clustering strategy."""
        pass
    
    @abstractmethod
    def get_results(self) -> ClusteringResult:
        """Get clustering results."""
        pass
    
    @abstractmethod
    def save_outputs(self, save_to_model_specs: bool = True) -> Dict[str, Path]:
        """
        Save clustering outputs to disk.
        
        Returns:
        --------
        output_files : dict
            Dictionary mapping output type to file path
        """
        pass
    
    def _create_2d_visualization(self, labels: np.ndarray, 
                                 dr_kwargs: Dict,
                                 dr_method: str = "umap",
                                 random_state: int = 42) -> None:
        """Create 2D visualization using specified dimensionality reduction method."""
        try:
            # Build DR parameters
            dr_params = {
                "n_components": 2,
                "random_state": random_state,
            }
            # Add method-specific params
            if dr_method in ["umap", "isomap", "spectral"]:
                dr_params["n_neighbors"] = dr_kwargs.get("n_neighbors", 10)
            if dr_method == "umap":
                dr_params["min_dist"] = max(0.01, dr_kwargs.get("min_dist", 0.0))
                dr_params["metric"] = "cosine"
            if dr_method == "tsne":
                dr_params["perplexity"] = dr_kwargs.get("perplexity", 30)
            
            reducer = get_reducer(dr_method, verbose=False, **dr_params)
            X2d = reducer.fit_transform(self.embeddings)
            
            # Save 2D coordinates for reuse in extensions
            coords_file = self.output_dir / f"{dr_method}_2d_coordinates.csv"
            coords_df = pd.DataFrame({
                'item_id': self.items['item_id'],
                f'{dr_method}_1': X2d[:, 0],
                f'{dr_method}_2': X2d[:, 1]
            })
            coords_df.to_csv(coords_file, index=False)
            
            # Create a simple 2D scatter plot
            fig, ax = plt.subplots(figsize=(12, 10))
            
            unique_clusters = sorted(np.unique(labels))
            colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_clusters))))
            
            # Plot each cluster
            for i, cluster_id in enumerate(unique_clusters):
                mask = labels == cluster_id
                color_idx = i % len(colors)
                label_text = f'Cluster {cluster_id}' if cluster_id >= 0 else 'Noise'
                ax.scatter(X2d[mask, 0], X2d[mask, 1], 
                          c=[colors[color_idx]], s=100, alpha=0.6, 
                          label=label_text, edgecolors='black', linewidth=0.5)
                
                # Add item labels
                for idx in np.where(mask)[0]:
                    ax.annotate(self.items['item_id'].iloc[idx], 
                               (X2d[idx, 0], X2d[idx, 1]),
                               fontsize=8, alpha=0.7)
            
            method_upper = dr_method.upper()
            ax.set_xlabel(f'{method_upper}-1', fontsize=12)
            ax.set_ylabel(f'{method_upper}-2', fontsize=12)
            ax.set_title(f'{method_upper} 2D Projection - Clusters', fontsize=14)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            viz_file = self.output_dir / f"{dr_method}_2d_clusters.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"      {method_upper} 2D plot saved to: {viz_file}")
            print(f"      {method_upper} 2D coordinates saved to: {coords_file}")
        except Exception as e:
            print(f"      Warning: {dr_method.upper()} 2D visualization failed: {e}")
    
    # Backward compatibility alias
    def _create_umap_2d_visualization(self, labels: np.ndarray, 
                                     umap_kwargs: Dict, 
                                     random_state: int = 42) -> None:
        """Create 2D UMAP visualization (backward compatibility wrapper)."""
        self._create_2d_visualization(labels, umap_kwargs, dr_method="umap", random_state=random_state)


# ============================================================================
# Pipeline A: KMeans Consensus Strategy (1000 runs)
# ============================================================================

class KMeansConsensusStrategy(ClusteringStrategy):
    """
    Pipeline A: KMeans with consensus clustering.
    
    Runs KMeans multiple times with different random seeds and creates
    a consensus clustering based on co-occurrence patterns.
    """
    
    def run(self) -> ClusteringResult:
        """Execute KMeans consensus clustering."""
        n_clusters = self.kwargs.get('n_clusters', 5)
        n_iterations = self.kwargs.get('n_iterations', 1000)
        umap_kwargs = self.kwargs.get('umap_kwargs', {
            "metric": "cosine",
            "n_neighbors": 10,
            "min_dist": 0.0,
            "n_components": 15,
        })
        linkage = self.kwargs.get('consensus_linkage', 'average')
        use_dr = self.kwargs.get('use_dr', True)
        dr_method = self.kwargs.get('dr_method', 'umap')
        
        n_items = len(self.items)
        validate_n_clusters(n_clusters, n_items)
        
        reduction_method = dr_method.upper() if use_dr else "No reduction"
        print(f"\n[Running] KMeans Consensus Strategy ({n_iterations} iterations, {reduction_method})...")
        
        # Storage for all runs
        all_results = []
        all_labels = []
        
        # Run with different random seeds
        for i in tqdm(range(n_iterations), desc="Running clustering"):
            try:
                # Optional dimensionality reduction
                if use_dr:
                    dr_params = {
                        "n_components": umap_kwargs.get("n_components", 15),
                        "random_state": i,
                    }
                    # Add method-specific params
                    if dr_method in ["umap", "isomap", "spectral"]:
                        dr_params["n_neighbors"] = umap_kwargs.get("n_neighbors", 10)
                    if dr_method == "umap":
                        dr_params["min_dist"] = umap_kwargs.get("min_dist", 0.0)
                        dr_params["metric"] = umap_kwargs.get("metric", "cosine")
                    if dr_method == "tsne":
                        dr_params["perplexity"] = umap_kwargs.get("perplexity", 30)
                    
                    reducer = get_reducer(dr_method, verbose=False, **dr_params)
                    Xr = reducer.fit_transform(self.embeddings)
                else:
                    # Use embeddings directly
                    Xr = self.embeddings
                
                # KMeans clustering
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    init='random',
                    n_init=1000,
                    random_state=i,
                    max_iter=300,
                )
                labels = kmeans.fit_predict(Xr)
                sil_score = silhouette_score(Xr, labels, metric='euclidean')
                
                all_results.append({
                    'run': i,
                    'inertia': kmeans.inertia_,
                    'silhouette': sil_score,
                })
                all_labels.append(labels)
                
            except Exception as e:
                print(f"\n      Warning: Run {i} failed: {e}")
                continue
        
        print(f"\n      Successfully completed {len(all_results)} runs")
        
        # Calculate stability metrics
        co_occurrence, ari_scores = calculate_cluster_stability(all_labels, n_items)
        stability_matrix = analyze_factor_composition_stability(
            all_labels, self.items['item_id'].values, self.output_dir
        )
        
        # Create consensus clustering
        consensus_labels, stability_scores = create_consensus_clustering(
            co_occurrence, n_clusters, linkage=linkage
        )
        
        # Validate consensus
        validation_metrics, ari_to_consensus = validate_consensus_clustering(
            consensus_labels, all_labels, co_occurrence
        )
        
        # Build result
        inertias = [r['inertia'] for r in all_results]
        silhouettes = [r['silhouette'] for r in all_results]
        
        diagnostics = {
            'all_results': all_results,
            'co_occurrence': co_occurrence,
            'stability_matrix': stability_matrix,
            'ari_scores': ari_scores,
            'ari_to_consensus': ari_to_consensus,
            'inertias': inertias,
            'silhouettes': silhouettes,
            'validation_metrics': validation_metrics,
            'n_successful_runs': len(all_results),
        }
        
        summary = {
            "model": self.model_name,
            "timestamp": self.timestamp,
            "n_iterations": n_iterations,
            "n_successful_runs": len(all_results),
            "n_clusters": n_clusters,
            "kmeans_metrics": {
                "mean_inertia": float(np.mean(inertias)),
                "std_inertia": float(np.std(inertias)),
                "min_inertia": float(np.min(inertias)),
                "max_inertia": float(np.max(inertias)),
                "mean_silhouette": float(np.mean(silhouettes)),
                "std_silhouette": float(np.std(silhouettes)),
                "min_silhouette": float(np.min(silhouettes)),
                "max_silhouette": float(np.max(silhouettes)),
            },
            "stability_metrics": {
                "mean_pairwise_ari": float(np.mean(ari_scores)),
                "median_pairwise_ari": float(np.median(ari_scores)),
                "std_pairwise_ari": float(np.std(ari_scores)),
                "min_pairwise_ari": float(np.min(ari_scores)),
                "max_pairwise_ari": float(np.max(ari_scores)),
            },
            "dr_method": dr_method,
            "umap_params": umap_kwargs,
            "kmeans_params": {
                "n_clusters": n_clusters,
                "init": "random",
                "n_init": 1,
            },
            "consensus_validation": validation_metrics,
        }
        
        self.result = ClusteringResult(
            labels=consensus_labels,
            probabilities=stability_scores,
            params={'n_clusters': n_clusters, 'n_iterations': n_iterations, 'linkage': linkage},
            diagnostics=diagnostics,
            metadata=summary,
        )
        
        return self.result
    
    def get_results(self) -> ClusteringResult:
        """Get clustering results."""
        if self.result is None:
            raise ValueError("Must run clustering first")
        return self.result
    
    def save_outputs(self, save_to_model_specs: bool = True) -> Dict[str, Path]:
        """Save all outputs for KMeans consensus clustering."""
        if self.result is None:
            raise ValueError("Must run clustering first")
        
        output_files = {}
        
        # 1. Main cluster assignments (for compatibility with labeling script)
        cluster_df = pd.DataFrame({
            "item_id": self.items["item_id"],
            "cluster": self.result.labels,
            "probability": self.result.probabilities,
        })
        main_file = self.output_dir / "cluster_assignments.csv"
        cluster_df.to_csv(main_file, index=False)
        output_files['cluster_assignments'] = main_file
        
        # 2. Detailed consensus assignments
        consensus_file = self.output_dir / "consensus_cluster_assignments.csv"
        cluster_df.to_csv(consensus_file, index=False)
        output_files['consensus_assignments'] = consensus_file
        
        # 3. All run results
        results_df = pd.DataFrame(self.result.diagnostics['all_results'])
        results_file = self.output_dir / "stability_results.csv"
        results_df.to_csv(results_file, index=False)
        output_files['stability_results'] = results_file
        
        # 4. Co-occurrence matrix
        co_occurrence_df = pd.DataFrame(
            self.result.diagnostics['co_occurrence'],
            index=self.items['item_id'],
            columns=self.items['item_id']
        )
        co_occurrence_file = self.output_dir / "co_occurrence_matrix.csv"
        co_occurrence_df.to_csv(co_occurrence_file)
        output_files['co_occurrence'] = co_occurrence_file
        
        # 5. Stability matrix
        stability_matrix = self.result.diagnostics['stability_matrix']
        stability_df = pd.DataFrame(
            stability_matrix,
            index=self.items['item_id'],
            columns=[f'Factor_{i+1}' for i in range(stability_matrix.shape[1])]
        )
        stability_file = self.output_dir / "factor_stability_matrix.csv"
        stability_df.to_csv(stability_file)
        output_files['stability_matrix'] = stability_file
        
        # 6. Summary JSON
        summary_file = self.output_dir / "stability_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(self.result.metadata, f, indent=2, ensure_ascii=False)
        output_files['summary'] = summary_file
        
        # 7. Visualizations
        plot_stability_results(
            self.result.diagnostics['inertias'],
            self.result.diagnostics['silhouettes'],
            self.result.diagnostics['co_occurrence'],
            self.result.diagnostics['stability_matrix'],
            self.items['item_id'].values,
            self.result.diagnostics['ari_scores'],
            self.result.diagnostics['ari_to_consensus'],
            self.output_dir
        )
        
        plot_consensus_model(
            self.result.labels,
            self.result.probabilities,
            self.items['item_id'].values,
            self.output_dir
        )
        
        # 8. 2D visualization
        umap_kwargs = self.result.metadata['umap_params']
        dr_method = self.result.metadata.get('dr_method', 'umap')
        self._create_2d_visualization(self.result.labels, umap_kwargs, dr_method=dr_method)
        output_files['viz_2d'] = self.output_dir / f"{dr_method}_2d_clusters.png"
        
        # 9. Save to model_specs.yaml
        if save_to_model_specs:
            from pathlib import Path
            root = Path(__file__).resolve().parents[2]
            spec_name, model_specs_path = save_consensus_to_model_specs(
                self.result.labels,
                self.result.probabilities,
                self.items,
                self.model_name,
                self.timestamp,
                umap_kwargs,
                self.result.params['n_iterations'],
                self.result.metadata,
                root=root
            )
            output_files['model_specs'] = model_specs_path
            print(f"      Consensus model saved as '{spec_name}' in: {model_specs_path}")
        
        return output_files


# ============================================================================
# Pipeline B: Similarity-based clustering (item-factor assignments)
# ============================================================================

class SimilarityClusteringStrategy(ClusteringStrategy):
    """
    Pipeline B: Similarity-based clustering using factor definitions.
    
    Assigns items to factors based on their cosine similarity to factor embeddings
    created from construct definitions in construct_definition.yaml.
    """
    
    def run(self) -> ClusteringResult:
        """Execute similarity-based clustering using factor embeddings."""
        construct_source = self.kwargs.get('construct_source', 'oxford')
        model_name = self.kwargs.get('model_name', self.model_name)
        min_similarity = self.kwargs.get('min_similarity', 0.0)
        
        print(f"\n[Running] Similarity-Based Strategy...")
        print(f"      Construct source: {construct_source}")
        print(f"      Model: {model_name}")
        print(f"      Min similarity threshold: {min_similarity}")
        
        # Load construct definitions
        factor_definitions = self._load_construct_definitions(construct_source)
        print(f"      Loaded {len(factor_definitions)} factor definitions")
        
        # Create factor embeddings
        factor_embeddings, factor_names = self._create_factor_embeddings(
            factor_definitions, model_name
        )
        print(f"      Created embeddings for {len(factor_names)} factors: {factor_names}")
        
        # Calculate similarity matrix: items x factors
        similarity_matrix = self._calculate_similarity_matrix(
            self.embeddings, factor_embeddings
        )
        
        # Assign items to most similar factor
        labels = np.argmax(similarity_matrix, axis=1)
        probabilities = np.max(similarity_matrix, axis=1)
        
        # Apply minimum similarity threshold (assign -1 if below threshold)
        labels[probabilities < min_similarity] = -1
        
        n_noise = np.sum(labels == -1)
        n_assigned = len(labels) - n_noise
        
        print(f"\n      Assignments complete:")
        print(f"        - Items assigned: {n_assigned}")
        print(f"        - Items below threshold: {n_noise}")
        print(f"        - Mean similarity: {probabilities[labels >= 0].mean():.3f}")
        
        # Build diagnostics
        diagnostics = {
            'similarity_matrix': similarity_matrix,
            'factor_names': factor_names,
            'factor_definitions': factor_definitions,
            'factor_embeddings': factor_embeddings,
        }
        
        # Build metadata summary (convert numpy types to Python native types)
        summary = {
            "model": self.model_name,
            "timestamp": self.timestamp,
            "n_factors": int(len(factor_names)),
            "construct_source": construct_source,
            "factor_names": [str(n) for n in factor_names],
            "similarity_metrics": {
                "mean_similarity": float(probabilities.mean()),
                "median_similarity": float(np.median(probabilities)),
                "min_similarity": float(probabilities.min()),
                "max_similarity": float(probabilities.max()),
                "mean_assigned_similarity": float(probabilities[labels >= 0].mean()) if n_assigned > 0 else 0.0,
            },
            "assignment_counts": {
                str(name): int(np.sum(labels == i))
                for i, name in enumerate(factor_names)
            },
            "n_noise": int(n_noise),
            "n_assigned": int(n_assigned),
            "min_similarity_threshold": float(min_similarity),
        }
        
        self.result = ClusteringResult(
            labels=labels,
            probabilities=probabilities,
            params={
                'construct_source': construct_source,
                'min_similarity': min_similarity,
                'n_factors': len(factor_names),
            },
            diagnostics=diagnostics,
            metadata=summary,
        )
        
        return self.result
    
    def _load_construct_definitions(self, construct_source: str) -> Dict[str, List[str]]:
        """Load construct definitions from YAML file."""
        # Find config directory
        root = Path(__file__).resolve().parents[2]
        config_path = root / "config" / "construct_definition.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Construct definition file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            all_definitions = yaml.safe_load(f)
        
        if construct_source not in all_definitions:
            available = list(all_definitions.keys())
            raise ValueError(
                f"Construct source '{construct_source}' not found. "
                f"Available sources: {available}"
            )
        
        # Get definitions for the specified source
        source_definitions = all_definitions[construct_source]
        
        # Combine all descriptions for each factor
        factor_definitions = {}
        for factor_name, descriptions in source_definitions.items():
            if descriptions is None:
                continue
            # Combine all descriptions into one text
            combined_text = " ".join(
                item['description'] 
                for item in descriptions 
                if isinstance(item, dict) and 'description' in item
            )
            if combined_text.strip():
                factor_definitions[factor_name] = combined_text
        
        return factor_definitions
    
    def _create_factor_embeddings(self, factor_definitions: Dict[str, str], 
                                   model_name: str) -> Tuple[np.ndarray, List[str]]:
        """Create embeddings for factor definitions."""
        from .embeddings import load_model
        
        # Load the same model used for items
        print(f"      Loading embedding model: {model_name}")
        model = load_model(model_name)
        
        factor_names = sorted(factor_definitions.keys())
        factor_texts = [factor_definitions[name] for name in factor_names]
        
        # Create embeddings
        if isinstance(model, dict) and model.get("type") == "sentence-transformers":
            st_model = model["model"]
            embeddings = st_model.encode(
                factor_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        elif isinstance(model, dict) and model.get("type") == "gpt":
            # Handle GPT models if needed
            from .embeddings import _embed_with_gpt
            embeddings = _embed_with_gpt(factor_texts, model["tokenizer"], model["model"])
        else:
            # Legacy support
            embeddings = model.encode(
                factor_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        
        return embeddings, factor_names
    
    def _calculate_similarity_matrix(self, item_embeddings: np.ndarray, 
                                     factor_embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between items and factors."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Compute similarity: items x factors
        # Both embeddings should already be normalized
        similarity = cosine_similarity(item_embeddings, factor_embeddings)
        
        return similarity
    
    def get_results(self) -> ClusteringResult:
        """Get clustering results."""
        if self.result is None:
            raise ValueError("Must run clustering first")
        return self.result
    
    def save_outputs(self, save_to_model_specs: bool = True) -> Dict[str, Path]:
        """Save all outputs for similarity-based clustering."""
        if self.result is None:
            raise ValueError("Must run clustering first")
        
        output_files = {}
        
        # 1. Main cluster assignments (for compatibility with labeling script)
        cluster_df = pd.DataFrame({
            "item_id": self.items["item_id"],
            "cluster": self.result.labels,
            "probability": self.result.probabilities,
        })
        main_file = self.output_dir / "cluster_assignments.csv"
        cluster_df.to_csv(main_file, index=False)
        output_files['cluster_assignments'] = main_file
        
        # 2. Detailed similarity scores for all factors
        factor_names = self.result.diagnostics['factor_names']
        similarity_matrix = self.result.diagnostics['similarity_matrix']
        
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=self.items['item_id'],
            columns=factor_names
        )
        similarity_file = self.output_dir / "item_factor_similarities.csv"
        similarity_df.to_csv(similarity_file)
        output_files['similarity_matrix'] = similarity_file
        
        # 3. Factor definitions used
        factor_definitions = self.result.diagnostics['factor_definitions']
        definitions_df = pd.DataFrame([
            {"factor": name, "definition": text}
            for name, text in factor_definitions.items()
        ])
        definitions_file = self.output_dir / "factor_definitions.csv"
        definitions_df.to_csv(definitions_file, index=False)
        output_files['factor_definitions'] = definitions_file
        
        # 4. Summary JSON
        summary_file = self.output_dir / "similarity_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(self.result.metadata, f, indent=2, ensure_ascii=False)
        output_files['summary'] = summary_file
        
        # 5. Visualizations
        self._plot_similarity_heatmap()
        output_files['similarity_heatmap'] = self.output_dir / "similarity_heatmap.png"
        
        self._plot_factor_assignments()
        output_files['factor_assignments'] = self.output_dir / "factor_assignments.png"
        
        # 6. UMAP 2D visualization (if applicable)
        # Use basic UMAP params for visualization
        umap_kwargs = {
            "metric": "cosine",
            "n_neighbors": 10,
            "min_dist": 0.0,
            "n_components": 2,
        }
        self._create_umap_2d_visualization(self.result.labels, umap_kwargs)
        output_files['umap_2d'] = self.output_dir / "umap_2d_clusters.png"
        
        # 7. Save to model_specs.yaml if requested
        if save_to_model_specs:
            spec_name, model_specs_path = self._save_to_model_specs()
            output_files['model_specs'] = model_specs_path
            print(f"      Similarity model saved as '{spec_name}' in: {model_specs_path}")
        
        return output_files
    
    def _plot_similarity_heatmap(self) -> None:
        """Plot heatmap of item-factor similarities."""
        similarity_matrix = self.result.diagnostics['similarity_matrix']
        factor_names = self.result.diagnostics['factor_names']
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(self.items) * 0.15)))
        
        im = ax.imshow(similarity_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xlabel('Factor', fontsize=12)
        ax.set_ylabel('Item Index', fontsize=12)
        ax.set_title('Item-to-Factor Cosine Similarity\n'
                     '(Higher values = stronger semantic alignment)', 
                     fontsize=14)
        
        # Set ticks
        ax.set_xticks(range(len(factor_names)))
        ax.set_xticklabels(factor_names, rotation=45, ha='right', fontsize=10)
        ax.set_yticks(np.arange(0, len(self.items), max(1, len(self.items) // 20)))
        ax.set_yticklabels(np.arange(0, len(self.items), max(1, len(self.items) // 20)), fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine Similarity', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'similarity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      Similarity heatmap saved to: {self.output_dir / 'similarity_heatmap.png'}")
    
    def _plot_factor_assignments(self) -> None:
        """Plot factor assignment distribution."""
        factor_names = self.result.diagnostics['factor_names']
        labels = self.result.labels
        probabilities = self.result.probabilities
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Assignment counts
        assigned_labels = labels[labels >= 0]
        unique_labels, counts = np.unique(assigned_labels, return_counts=True)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(factor_names)))
        bars = ax1.bar(range(len(unique_labels)), counts, color=[colors[i] for i in unique_labels])
        
        ax1.set_xlabel('Factor', fontsize=12)
        ax1.set_ylabel('Number of Items', fontsize=12)
        ax1.set_title(f'Factor Assignment Counts\n'
                      f'{len(assigned_labels)} items assigned to {len(unique_labels)} factors',
                      fontsize=14)
        ax1.set_xticks(range(len(unique_labels)))
        ax1.set_xticklabels([factor_names[i] for i in unique_labels], rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom', fontsize=10)
        
        # Right: Similarity distribution by factor
        for i, name in enumerate(factor_names):
            mask = labels == i
            if np.any(mask):
                sims = probabilities[mask]
                ax2.hist(sims, bins=20, alpha=0.5, label=name, color=colors[i])
        
        ax2.set_xlabel('Similarity Score', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Similarity Scores by Factor', fontsize=14)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'factor_assignments.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      Factor assignments plot saved to: {self.output_dir / 'factor_assignments.png'}")
    
    def _save_to_model_specs(self) -> Tuple[str, Path]:
        """Save the similarity-based model to model_specs.yaml."""
        root = Path(__file__).resolve().parents[2]
        config_dir = root / "config"
        model_specs_path = config_dir / "model_specs.yaml"
        
        # Load existing model specs
        if model_specs_path.exists():
            with open(model_specs_path, 'r', encoding='utf-8') as f:
                model_specs = yaml.safe_load(f) or {}
        else:
            model_specs = {}
        
        factor_names = self.result.diagnostics['factor_names']
        labels = self.result.labels
        probabilities = self.result.probabilities
        
        # Group items by factor
        factor_indices = {}
        factor_items = {}
        factor_sizes = {}
        factor_similarity_scores = {}
        
        for i, factor_name in enumerate(factor_names):
            mask = labels == i
            if not np.any(mask):
                continue
            
            assigned_items = self.items[mask]
            assigned_indices = assigned_items.index.tolist()
            assigned_ids = assigned_items['item_id'].tolist()
            assigned_probs = probabilities[mask].tolist()
            
            factor_key = f"factor_{i+1}"
            factor_indices[factor_key] = assigned_indices
            factor_items[factor_key] = assigned_ids
            factor_sizes[factor_key] = len(assigned_ids)
            factor_similarity_scores[factor_key] = {
                "mean": float(np.mean(assigned_probs)),
                "median": float(np.median(assigned_probs)),
                "min": float(np.min(assigned_probs)),
                "max": float(np.max(assigned_probs)),
            }
        
        # Create model spec entry (convert numpy types to Python native types)
        model_slug = self.model_name.replace("/", "_").replace("-", "_")
        construct_source = self.result.params['construct_source']
        spec_name = f"{model_slug}_similarity_{construct_source}"
        
        n_assigned = int(np.sum(labels >= 0))
        n_noise = int(np.sum(labels == -1))
        
        model_specs[spec_name] = {
            "source": "similarity_clustering",
            "model": self.model_name,
            "date": self.timestamp.split("_")[0],
            "n_factors": int(len(factor_names)),
            "n_items": int(len(self.items)),
            "n_assigned": n_assigned,
            "n_noise": n_noise,
            "method": "Similarity to factor embeddings",
            "construct_source": construct_source,
            "factor_names": [str(n) for n in factor_names],
            "factor_indices": {k: [int(x) for x in v] for k, v in factor_indices.items()},
            "factor_items": {k: [str(x) for x in v] for k, v in factor_items.items()},
            "factor_sizes": {k: int(v) for k, v in factor_sizes.items()},
            "factor_similarity_scores": factor_similarity_scores,
            "similarity_metrics": self.result.metadata["similarity_metrics"],
            "min_similarity_threshold": float(self.result.params['min_similarity']),
            "timestamp": self.timestamp,
            "description": (
                f"Similarity-based clustering using '{construct_source}' construct definitions. "
                f"Items are assigned to factors based on cosine similarity to factor embeddings "
                f"created from theoretical construct definitions. {n_assigned} items assigned "
                f"across {int(len(factor_names))} factors: {', '.join([str(n) for n in factor_names])}. "
                f"This approach provides theory-driven factor assignments."
            ),
        }
        
        # Save back to YAML
        with open(model_specs_path, "w", encoding="utf-8") as f:
            yaml.dump(model_specs, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        return spec_name, model_specs_path

# ============================================================================
# Pipeline C: KMeans-simple (1000 runs)
# ============================================================================

class KMeansSimpleStrategy(ClusteringStrategy):
    """
    Pipeline A: KMeans WITHOUT consensus clustering.
    
    Runs KMeans with high n_init.
    """
    
    def run(self) -> ClusteringResult:
        """Execute KMeans simple clustering."""
        n_clusters = self.kwargs.get('n_clusters', 5)
        n_iterations = self.kwargs.get('n_iterations', 1000)
        umap_kwargs = self.kwargs.get('umap_kwargs', {
            "metric": "cosine",
            "n_neighbors": 10,
            "min_dist": 0.0,
            "n_components": 15,
        })
        use_dr = self.kwargs.get('use_dr', True)
        dr_method = self.kwargs.get('dr_method', 'umap')
        
        n_items = len(self.items)
        validate_n_clusters(n_clusters, n_items)
        
        reduction_method = dr_method.upper() if use_dr else "No reduction"
        print(f"\n[Running] KMeans Simple Strategy ({n_iterations} iterations, {reduction_method})...")
        
        # Optional dimensionality reduction (shared init)
        if use_dr:
            dr_params = {
                "n_components": umap_kwargs.get("n_components", 15),
                "random_state": 0,
            }
            # Add method-specific params
            if dr_method in ["umap", "isomap", "spectral"]:
                dr_params["n_neighbors"] = umap_kwargs.get("n_neighbors", 10)
            if dr_method == "umap":
                dr_params["min_dist"] = umap_kwargs.get("min_dist", 0.0)
                dr_params["metric"] = umap_kwargs.get("metric", "cosine")
            if dr_method == "tsne":
                dr_params["perplexity"] = umap_kwargs.get("perplexity", 30)
            
            reducer = get_reducer(dr_method, verbose=False, **dr_params)
            Xr = reducer.fit_transform(self.embeddings)
        else:
            # Use embeddings directly
            Xr = self.embeddings

        # KMeans with many initializations (n_init)
        kmeans = KMeans(
            n_clusters=n_clusters,
            init='random',
            n_init=n_iterations,
            random_state=0,
            max_iter=300,
        )
        labels = kmeans.fit_predict(Xr)

        # Confidence proxy: inverse distance to assigned centroid
        distances = kmeans.transform(Xr)
        min_distances = distances.min(axis=1)
        probabilities = 1.0 / (1.0 + min_distances)

        sil_score = silhouette_score(Xr, labels, metric='euclidean') if n_clusters > 1 else float("nan")

        diagnostics = {
            'all_results': [{'inertia': kmeans.inertia_, 'silhouette': sil_score}],
            'inertia': [kmeans.inertia_],
            'silhouette': [sil_score],
        }

        summary = {
            "model": self.model_name,
            "timestamp": self.timestamp,
            "n_iterations": n_iterations,
            "n_successful_runs": 1,
            "n_clusters": n_clusters,
            "kmeans_metrics": {
                "mean_inertia": float(kmeans.inertia_),
                "std_inertia": 0.0,
                "min_inertia": float(kmeans.inertia_),
                "max_inertia": float(kmeans.inertia_),
                "mean_silhouette": float(sil_score),
                "std_silhouette": 0.0,
                "min_silhouette": float(sil_score),
                "max_silhouette": float(sil_score),
            },
            "dr_method": dr_method,
            "umap_params": umap_kwargs,
            "kmeans_params": {
                "n_clusters": n_clusters,
                "init": "random",
                "n_init": n_iterations,
            }
        }

        self.result = ClusteringResult(
            labels=labels,
            probabilities=probabilities,
            params={'n_clusters': n_clusters, 'n_iterations': n_iterations},
            diagnostics=diagnostics,
            metadata=summary,
        )

        print(f"\n      Successfully completed {n_iterations} KMeans initializations")
        print(f"      Inertia: {kmeans.inertia_:.2f}")
        print(f"      Silhouette: {sil_score:.4f}")

        return self.result
    
    def get_results(self) -> ClusteringResult:
        """Get clustering results."""
        if self.result is None:
            raise ValueError("Must run clustering first")
        return self.result
    
    def save_outputs(self, save_to_model_specs: bool = True) -> Dict[str, Path]:
        """Save outputs for KMeans-simple clustering."""
        if self.result is None:
            raise ValueError("Must run clustering first")
        
        output_files = {}
        
        # 1. Main cluster assignments (compatible with labeling script)
        cluster_df = pd.DataFrame({
            "item_id": self.items["item_id"],
            "cluster": self.result.labels,
            "probability": self.result.probabilities,
        })
        main_file = self.output_dir / "cluster_assignments.csv"
        cluster_df.to_csv(main_file, index=False)
        output_files['cluster_assignments'] = main_file

        # 2. Run summary
        results_df = pd.DataFrame(self.result.diagnostics['all_results'])
        results_file = self.output_dir / "kmeans_simple_results.csv"
        results_df.to_csv(results_file, index=False)
        output_files['results'] = results_file

        # 3. Summary JSON
        summary_file = self.output_dir / "kmeans_simple_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(self.result.metadata, f, indent=2, ensure_ascii=False)
        output_files['summary'] = summary_file

        # 4. 2D visualization
        umap_kwargs = self.result.metadata['umap_params']
        dr_method = self.result.metadata.get('dr_method', 'umap')
        self._create_2d_visualization(self.result.labels, umap_kwargs, dr_method=dr_method)
        output_files['viz_2d'] = self.output_dir / f"{dr_method}_2d_clusters.png"

        return output_files

# ============================================================================
# Pipeline D: HDBSCAN Strategy (Legacy)
# ============================================================================

class HDBSCANStrategy(ClusteringStrategy):
    """
    Pipeline D: HDBSCAN density-based clustering.
    
    Uses dimensionality reduction + HDBSCAN for model-free cluster discovery.
    """
    
    def run(self) -> ClusteringResult:
        """Execute HDBSCAN clustering."""
        umap_kwargs = self.kwargs.get('umap_kwargs', {
            "metric": "cosine",
            "n_neighbors": 10,
            "min_dist": 0.0,
            "n_components": 15,
        })
        hdbscan_grid = self.kwargs.get('hdbscan_param_grid', {
            "min_cluster_size": [2, 3, 4, 5, 6],
            "min_samples": [None, 1, 2],
        })
        dr_method = self.kwargs.get('dr_method', 'umap')
        
        print(f"\n[Running] HDBSCAN Strategy ({dr_method.upper()} reduction)...")
        
        # Run dimensionality reduction + HDBSCAN
        result = umap_hdbscan_auto(
            self.embeddings,
            umap_kwargs=umap_kwargs,
            hdbscan_param_grid=hdbscan_grid,
            random_state=42,
            use_dr=True,
            dr_method=dr_method,
        )
        
        # Build ClusteringResult
        self.result = ClusteringResult(
            labels=result.labels,
            probabilities=result.probabilities,
            params=result.params,
            diagnostics=result.diagnostics,
            metadata={
                "model": self.model_name,
                "timestamp": self.timestamp,
                "method": "HDBSCAN",
                "dr_method": dr_method,
                "umap_params": umap_kwargs,
                "hdbscan_grid": hdbscan_grid,
            }
        )
        
        return self.result
    
    def get_results(self) -> ClusteringResult:
        """Get clustering results."""
        if self.result is None:
            raise ValueError("Must run clustering first")
        return self.result
    
    def save_outputs(self, save_to_model_specs: bool = False) -> Dict[str, Path]:
        """Save HDBSCAN clustering outputs."""
        if self.result is None:
            raise ValueError("Must run clustering first")
        
        output_files = {}
        
        # 1. Cluster assignments
        cluster_df = pd.DataFrame({
            "item_id": self.items["item_id"],
            "cluster": self.result.labels,
            "probability": self.result.probabilities,
        })
        main_file = self.output_dir / "cluster_assignments.csv"
        cluster_df.to_csv(main_file, index=False)
        output_files['cluster_assignments'] = main_file
        
        # 2. Diagnostics JSON
        diagnostics_file = self.output_dir / "cluster_diagnostics.json"
        # Convert numpy types to Python types for JSON serialization
        diagnostics_serializable = {}
        for k, v in self.result.diagnostics.items():
            if isinstance(v, (np.integer, np.floating)):
                diagnostics_serializable[k] = float(v)
            elif isinstance(v, dict):
                diagnostics_serializable[k] = {
                    kk: float(vv) if isinstance(vv, (np.integer, np.floating)) else vv
                    for kk, vv in v.items()
                }
            else:
                diagnostics_serializable[k] = v
        
        with open(diagnostics_file, "w", encoding="utf-8") as f:
            json.dump(diagnostics_serializable, f, indent=2, ensure_ascii=False)
        output_files['diagnostics'] = diagnostics_file
        
        # 3. 2D visualization
        umap_kwargs = self.result.metadata['umap_params']
        dr_method = self.result.metadata.get('dr_method', 'umap')
        self._create_2d_visualization(self.result.labels, umap_kwargs, dr_method=dr_method)
        output_files['viz_2d'] = self.output_dir / f"{dr_method}_2d_clusters.png"
        
        return output_files