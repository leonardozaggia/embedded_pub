"""
Visualization utilities for extended clustering analysis.

Functions for creating enhanced visualizations showing:
1. Newly assigned items with red contours
2. Items that changed cluster assignment after extension
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def create_extended_visualization(
    coords_2d: pd.DataFrame,
    labels_combined: np.ndarray,
    is_reference: np.ndarray,
    items_combined: pd.DataFrame,
    output_dir: Path,
    dr_method: str = "umap"
):
    """
    Create 2D visualization showing reference items and newly assigned items.
    
    Parameters:
    -----------
    coords_2d : pd.DataFrame
        DataFrame with item_id and 2D coordinates (e.g., umap_1, umap_2)
    labels_combined : np.ndarray
        Cluster assignments for all items (reference + new)
    is_reference : np.ndarray
        Boolean array indicating which items are reference items
    items_combined : pd.DataFrame
        Combined items dataframe with item_id and item_text
    output_dir : Path
        Output directory for saving plot
    dr_method : str
        Dimensionality reduction method ('umap', 'tsne', 'pca')
    
    Returns:
    --------
    viz_file : Path
        Path to saved visualization
    """
    print(f"\n      Creating enhanced {dr_method.upper()} 2D visualization...")
    
    # Extract coordinates
    X2d = coords_2d[[f'{dr_method}_1', f'{dr_method}_2']].values
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 12))
    
    unique_clusters = sorted(np.unique(labels_combined[labels_combined >= 0]))
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_clusters))))
    
    # Plot each cluster
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = labels_combined == cluster_id
        color_idx = i % len(colors)
        
        # Separate reference and new items
        ref_mask = cluster_mask & is_reference
        new_mask = cluster_mask & ~is_reference
        
        # Plot reference items (no red contour)
        if ref_mask.sum() > 0:
            ax.scatter(X2d[ref_mask, 0], X2d[ref_mask, 1],
                      c=[colors[color_idx]], s=120, alpha=0.7,
                      edgecolors='black', linewidth=1.0,
                      label=f'Cluster {cluster_id}' if i == 0 or not new_mask.any() else "")
        
        # Plot new items (with red contour)
        if new_mask.sum() > 0:
            ax.scatter(X2d[new_mask, 0], X2d[new_mask, 1],
                      c=[colors[color_idx]], s=150, alpha=0.7,
                      edgecolors='red', linewidth=2.5,
                      label=f'Cluster {cluster_id} (new)' if i == 0 else "")
        
        # Add item labels
        for idx in np.where(cluster_mask)[0]:
            fontweight = 'bold' if not is_reference[idx] else 'normal'
            text_color = 'red' if not is_reference[idx] else 'black'
            ax.annotate(items_combined['item_id'].iloc[idx],
                       (X2d[idx, 0], X2d[idx, 1]),
                       fontsize=8, alpha=0.8, 
                       fontweight=fontweight,
                       color=text_color)
    
    # Create custom legend
    legend_elements = []
    for i, cluster_id in enumerate(unique_clusters):
        color_idx = i % len(colors)
        legend_elements.append(
            mpatches.Patch(facecolor=colors[color_idx], 
                          edgecolor='black', 
                          label=f'Cluster {cluster_id}')
        )
    
    legend_elements.append(
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='gray', markersize=10,
                  markeredgecolor='black', markeredgewidth=1.0,
                  label='Reference items', linestyle='')
    )
    legend_elements.append(
        plt.Line2D([0], [0], marker='o', color='w',
                  markerfacecolor='gray', markersize=12,
                  markeredgecolor='red', markeredgewidth=2.5,
                  label='Newly assigned items', linestyle='')
    )
    
    method_upper = dr_method.upper()
    ax.set_xlabel(f'{method_upper}-1', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'{method_upper}-2', fontsize=13, fontweight='bold')
    ax.set_title(f'{method_upper} 2D Projection - Extended Clustering\n'
                f'Reference items (n={is_reference.sum()}) + '
                f'New items (n={(~is_reference).sum()})',
                fontsize=15, fontweight='bold')
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
             loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    viz_file = output_dir / f"{dr_method}_2d_extended_clusters.png"
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"      Enhanced {method_upper} plot saved: {viz_file}")
    
    return viz_file


def create_cluster_change_visualization(
    coords_2d: pd.DataFrame,
    labels_original: np.ndarray,
    labels_new: np.ndarray,
    items_df: pd.DataFrame,
    output_dir: Path,
    dr_method: str = "umap"
):
    """
    Create visualization highlighting items that changed cluster assignment.
    
    Parameters:
    -----------
    coords_2d : pd.DataFrame
        DataFrame with item_id and 2D coordinates
    labels_original : np.ndarray
        Original cluster assignments (before extension)
    labels_new : np.ndarray
        New cluster assignments (after extension)
    items_df : pd.DataFrame
        Items dataframe with item_id
    output_dir : Path
        Output directory for saving plot
    dr_method : str
        Dimensionality reduction method
    
    Returns:
    --------
    viz_file : Path
        Path to saved visualization
    """
    print(f"\n      Creating cluster change visualization...")
    
    # Extract coordinates
    X2d = coords_2d[[f'{dr_method}_1', f'{dr_method}_2']].values
    
    # Identify items that changed
    changed_mask = labels_original != labels_new
    n_changed = changed_mask.sum()
    
    print(f"      Found {n_changed} items that changed clusters")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 12))
    
    unique_clusters = sorted(np.unique(labels_new[labels_new >= 0]))
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_clusters))))
    
    # Plot each cluster with new assignments
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = labels_new == cluster_id
        color_idx = i % len(colors)
        
        # Separate changed and unchanged items
        unchanged_mask = cluster_mask & ~changed_mask
        changed_in_cluster = cluster_mask & changed_mask
        
        # Plot unchanged items
        if unchanged_mask.sum() > 0:
            ax.scatter(X2d[unchanged_mask, 0], X2d[unchanged_mask, 1],
                      c=[colors[color_idx]], s=120, alpha=0.7,
                      marker='o',
                      edgecolors='black', linewidth=1.0,
                      label=f'Cluster {cluster_id}' if i == 0 else "")
        
        # Plot changed items (star marker with purple edge)
        if changed_in_cluster.sum() > 0:
            ax.scatter(X2d[changed_in_cluster, 0], X2d[changed_in_cluster, 1],
                      c=[colors[color_idx]], s=250, alpha=0.9,
                      marker='*',
                      edgecolors='purple', linewidth=3.0)
        
        # Add item labels
        for idx in np.where(cluster_mask)[0]:
            if changed_mask[idx]:
                # Changed items: bold purple text with cluster change info
                orig_cluster = labels_original[idx]
                ax.annotate(f"{items_df['item_id'].iloc[idx]}\n({orig_cluster}→{cluster_id})",
                           (X2d[idx, 0], X2d[idx, 1]),
                           fontsize=9, alpha=0.95, 
                           fontweight='bold',
                           color='purple',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
            else:
                # Unchanged items: normal text
                ax.annotate(items_df['item_id'].iloc[idx],
                           (X2d[idx, 0], X2d[idx, 1]),
                           fontsize=8, alpha=0.7)
    
    # Create custom legend
    legend_elements = []
    for i, cluster_id in enumerate(unique_clusters):
        color_idx = i % len(colors)
        legend_elements.append(
            mpatches.Patch(facecolor=colors[color_idx], 
                          edgecolor='black', 
                          label=f'Cluster {cluster_id}')
        )
    
    legend_elements.append(
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='gray', markersize=10,
                  markeredgecolor='black', markeredgewidth=1.0,
                  label='Unchanged assignment', linestyle='')
    )
    legend_elements.append(
        plt.Line2D([0], [0], marker='*', color='w',
                  markerfacecolor='gray', markersize=15,
                  markeredgecolor='purple', markeredgewidth=3.0,
                  label=f'Changed cluster (n={n_changed})', linestyle='')
    )
    
    method_upper = dr_method.upper()
    ax.set_xlabel(f'{method_upper}-1', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'{method_upper}-2', fontsize=13, fontweight='bold')
    ax.set_title(f'{method_upper} 2D Projection - Cluster Changes After Extension\n'
                f'{n_changed} items changed clusters (marked with ★)',
                fontsize=15, fontweight='bold')
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
             loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    viz_file = output_dir / f"{dr_method}_2d_cluster_changes.png"
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"      Cluster changes plot saved: {viz_file}")
    print(f"      Items that changed: {n_changed} / {len(labels_original)}")
    
    # Print detailed change summary
    if n_changed > 0:
        print(f"\n      Cluster transition summary:")
        for idx in np.where(changed_mask)[0]:
            item_id = items_df['item_id'].iloc[idx]
            orig = labels_original[idx]
            new = labels_new[idx]
            print(f"        {item_id}: Cluster {orig} → Cluster {new}")
    
    return viz_file
