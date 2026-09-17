"""
Visualization utilities for cluster centroid labeling.

This module provides functions to visualize how cluster centroids relate to
individual items and cognitive concepts.
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


def plot_centroid_heatmap(
    centroids: Dict[int, np.ndarray],
    embeddings: np.ndarray,
    labels: np.ndarray,
    item_ids: List[str],
    outpath: str,
    top_k: int = 10
):
    """
    Plot heatmap showing similarity of each item to each cluster centroid.
    
    Args:
        centroids: Dictionary mapping cluster_id -> centroid vector
        embeddings: All item embeddings
        labels: Cluster labels for each item
        item_ids: Item identifiers
        outpath: Path to save figure
        top_k: Number of top items to show per cluster
    """
    n_clusters = len(centroids)
    
    # Compute all similarities
    centroid_matrix = np.vstack([centroids[i] for i in sorted(centroids.keys())])
    similarities = cosine_similarity(embeddings, centroid_matrix)
    
    # Create dataframe
    sim_df = pd.DataFrame(
        similarities,
        index=item_ids,
        columns=[f'Cluster {i}' for i in sorted(centroids.keys())]
    )
    
    # Add true cluster labels
    sim_df['True Cluster'] = labels
    
    # Sort by true cluster and similarity
    sim_df = sim_df.sort_values(['True Cluster', sim_df.columns[0]], ascending=[True, False])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, max(10, len(item_ids) * 0.3)))
    
    # Plot heatmap
    sns.heatmap(
        sim_df.drop('True Cluster', axis=1),
        cmap='YlOrRd',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Cosine Similarity'},
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_title('Item Similarity to Cluster Centroids', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cluster Centroid', fontsize=12)
    ax.set_ylabel('Item', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Centroid heatmap saved to: {outpath}")


def plot_cluster_representatives(
    centroids: Dict[int, np.ndarray],
    embeddings: np.ndarray,
    items: pd.DataFrame,
    outpath: str,
    top_k: int = 5
):
    """
    Create bar plots showing the most representative items for each cluster.
    
    Args:
        centroids: Dictionary mapping cluster_id -> centroid vector
        embeddings: All item embeddings
        items: DataFrame with item metadata
        outpath: Path to save figure
        top_k: Number of top items to show per cluster
    """
    n_clusters = len(centroids)
    fig, axes = plt.subplots(
        n_clusters, 1,
        figsize=(12, 4 * n_clusters),
        squeeze=False
    )
    
    item_ids = items['item_id'].tolist()
    
    for idx, (cluster_id, centroid) in enumerate(sorted(centroids.items())):
        ax = axes[idx, 0]
        
        # Compute similarities
        similarities = cosine_similarity(centroid.reshape(1, -1), embeddings)[0]
        
        # Get top-k items
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_ids = [item_ids[i] for i in top_indices]
        top_sims = similarities[top_indices]
        
        # Get short titles
        top_titles = []
        for item_id in top_ids:
            title = items[items['item_id'] == item_id]['item_title'].values[0]
            if len(title) > 40:
                title = title[:37] + '...'
            top_titles.append(title)
        
        # Plot bars
        colors = plt.cm.viridis(top_sims)
        bars = ax.barh(range(top_k), top_sims, color=colors)
        
        # Labels
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([f"{id_}\n{title}" for id_, title in zip(top_ids, top_titles)], 
                           fontsize=9)
        ax.set_xlabel('Cosine Similarity to Centroid', fontsize=11)
        ax.set_title(f'Cluster {cluster_id}: Most Representative Items', 
                     fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, sim) in enumerate(zip(bars, top_sims)):
            ax.text(sim + 0.01, i, f'{sim:.3f}', 
                   va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Cluster representatives plot saved to: {outpath}")


def plot_concept_decoding(
    concept_results: pd.DataFrame,
    outpath: str
):
    """
    Visualize how cluster centroids map to cognitive concepts.
    
    Args:
        concept_results: DataFrame with concept decoding results
        outpath: Path to save figure
    """
    n_clusters = len(concept_results)
    
    fig, ax = plt.subplots(figsize=(10, max(6, n_clusters * 1.5)))
    
    # Extract top concepts and similarities
    cluster_ids = concept_results['cluster_id'].values
    top_concepts = concept_results['top_concept'].values
    top_sims = concept_results['top_concept_similarity'].values
    
    # Create horizontal bar plot
    colors = plt.cm.RdYlGn(top_sims)
    bars = ax.barh(range(n_clusters), top_sims, color=colors)
    
    # Labels
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels([f"Cluster {cid}" for cid in cluster_ids], fontsize=11)
    ax.set_xlabel('Similarity to Top Cognitive Concept', fontsize=12)
    ax.set_title('Cluster Centroids Decoded to Cognitive Domains', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(0.6, top_sims.max() + 0.1))
    ax.grid(axis='x', alpha=0.3)
    
    # Add concept labels on bars
    for i, (bar, concept, sim) in enumerate(zip(bars, top_concepts, top_sims)):
        ax.text(sim + 0.01, i, f'{concept} ({sim:.3f})', 
               va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Concept decoding plot saved to: {outpath}")


def create_all_visualizations(
    centroids: Dict[int, np.ndarray],
    embeddings: np.ndarray,
    labels: np.ndarray,
    items: pd.DataFrame,
    outdir: str,
    concept_results: Optional[pd.DataFrame] = None,
    top_k: int = 7
):
    """
    Generate all centroid labeling visualizations.
    
    Args:
        centroids: Dictionary mapping cluster_id -> centroid vector
        embeddings: All item embeddings
        labels: Cluster labels for each item
        items: DataFrame with item metadata
        outdir: Output directory
        concept_results: Optional DataFrame with concept decoding results
        top_k: Number of top items to show
    """
    os.makedirs(outdir, exist_ok=True)
    
    item_ids = items['item_id'].tolist()
    
    print("\nGenerating visualizations...")
    
    # 1. Heatmap of item-centroid similarities
    plot_centroid_heatmap(
        centroids, embeddings, labels, item_ids,
        outpath=os.path.join(outdir, 'centroid_similarity_heatmap.png'),
        top_k=top_k
    )
    
    # 2. Representative items bar plots
    plot_cluster_representatives(
        centroids, embeddings, items,
        outpath=os.path.join(outdir, 'cluster_representative_items.png'),
        top_k=top_k
    )
    
    # 3. Concept decoding (if available)
    if concept_results is not None:
        plot_concept_decoding(
            concept_results,
            outpath=os.path.join(outdir, 'concept_decoding.png')
        )
    
    print("✓ All visualizations created!")
