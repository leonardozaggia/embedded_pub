
# --- UTF-8 stdout/stderr: this script prints Unicode (arrows, checkmarks)
# which crashes under Windows' default cp1252 encoding; force UTF-8. ---
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")
del _stream

#!/usr/bin/env python
"""
Step 5: Compare Detected Model to Reference Models

Compares the data-driven clustering solution from Step 03 to:
1. Theoretical measurement model (expert-defined)
2. Empirical measurement model (normative sample CFA)

Computes comparison metrics:
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Item overlap/confusion matrices
- Factor alignment analysis

Usage:
    python 05_compare_models.py --model all-mpnet-base-v2
    python 05_compare_models.py --cluster-file data/outputs/main/b4_analysis/03_clustering/cluster_assignments.csv
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns


def load_cluster_labels(cluster_labels_file: Path) -> Tuple[List[str], Dict[int, str]]:
    """
    Load cluster labels from step 04 output.
    
    Looks for cluster_concept_labels.csv from step 04.
    
    Args:
        cluster_labels_file: Path to cluster_concept_labels.csv from step 04
        
    Returns:
        Tuple of (list of label names, dict mapping cluster_id to label)
    """
    labels_path = Path(cluster_labels_file)
    
    # Load the cluster concept labels file
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)
        
        # Check for cluster_concept_labels.csv format (cluster_id, top_concept)
        if 'cluster_id' in labels_df.columns and 'top_concept' in labels_df.columns:
            label_dict = dict(zip(labels_df['cluster_id'], labels_df['top_concept']))
            # Create abbreviated labels (first letter, capitalized)
            label_list = [label_dict.get(i, f"Cluster_{i}")[0].upper() if i in label_dict else f"C{i}"
                         for i in sorted(label_dict.keys())]
            return label_list, label_dict
        
        # Fallback: check for generic label/cluster columns
        elif 'label' in labels_df.columns and 'cluster' in labels_df.columns:
            label_dict = dict(zip(labels_df['cluster'], labels_df['label']))
            label_list = [label_dict.get(i, f"Cluster_{i}") 
                         for i in sorted(label_dict.keys())]
            return label_list, label_dict
    
    # Fallback to generic labels if file not found or parsing failed
    print(f"      Warning: Could not load labels from {labels_path}, using generic labels")
    # Assume 5 clusters as default (can be overridden later based on actual data)
    label_list = [f"Cluster_{i}" for i in range(5)]
    label_dict = {i: f"Cluster_{i}" for i in range(5)}
    
    return label_list, label_dict


def load_reference_models(yaml_path: Path) -> Dict:
    """Load reference measurement models from YAML file."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)



def convert_factor_dict_to_labels(factor_dict: Dict[str, List[str]], all_items: List[str]) -> np.ndarray:
    """
    Convert factor dictionary to integer labels array.
    
    Args:
        factor_dict: Dictionary mapping factor names to item lists
        all_items: Ordered list of all item IDs
        
    Returns:
        Array of integer labels (-1 for unassigned items)
    """
    labels = np.full(len(all_items), -1, dtype=int)
    
    for factor_idx, (factor_name, items) in enumerate(factor_dict.items()):
        for item in items:
            if item in all_items:
                item_idx = all_items.index(item)
                labels[item_idx] = factor_idx
    
    return labels


def convert_factor_dict_to_item_factor_map(factor_dict: Dict[str, List[str]], all_items: List[str]) -> Dict[str, List[int]]:
    """
    Build mapping of items to their factor indices, allowing multi-factor assignments.
    
    Args:
        factor_dict: Dictionary mapping factor names to item lists
        all_items: Ordered list of all item IDs
        
    Returns:
        Dict mapping item_id -> list of factor indices
    """
    item_to_factors = {item: [] for item in all_items}
    
    for factor_idx, (factor_name, items) in enumerate(factor_dict.items()):
        for item in items:
            if item in all_items:
                item_to_factors[item].append(factor_idx)
    
    return item_to_factors


def expand_labels_for_multi_factor(all_items: List[str],
                                   detected_labels: np.ndarray,
                                   item_to_factors: Dict[str, List[int]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Expand detected and reference labels to handle multi-factor items.
    
    Items that belong to multiple factors are duplicated in the output arrays.
    For example, if COG_035 belongs to both factor 2 and factor 3, it appears
    twice in the output.
    
    Args:
        all_items: Original list of all items
        detected_labels: Original detected cluster labels
        item_to_factors: Dict mapping item_id -> list of factor indices
        
    Returns:
        Tuple of (expanded_detected_labels, expanded_reference_labels, expanded_items)
    """
    expanded_detected = []
    expanded_reference = []
    expanded_items = []
    
    for item_idx, item_id in enumerate(all_items):
        factors = item_to_factors.get(item_id, [])
        
        if not factors:  # Unassigned items
            expanded_detected.append(detected_labels[item_idx])
            expanded_reference.append(-1)
            expanded_items.append(item_id)
        else:
            # Add this item once for each factor it belongs to
            for factor_idx in factors:
                expanded_detected.append(detected_labels[item_idx])
                expanded_reference.append(factor_idx)
                expanded_items.append(item_id)
    
    return np.array(expanded_detected), np.array(expanded_reference), expanded_items


def compute_alignment_metrics(detected_labels: np.ndarray, 
                              reference_labels: np.ndarray,
                              reference_name: str) -> Dict:
    """
    Compute alignment metrics between detected and reference models.
    
    Args:
        detected_labels: Labels from clustering algorithm
        reference_labels: Labels from reference model
        reference_name: Name of reference model (for reporting)
        
    Returns:
        Dictionary of metrics
    """
    # Filter out unassigned items (-1) from reference if any
    valid_mask = reference_labels != -1
    
    if not valid_mask.all():
        detected_filtered = detected_labels[valid_mask]
        reference_filtered = reference_labels[valid_mask]
        n_excluded = (~valid_mask).sum()
    else:
        detected_filtered = detected_labels
        reference_filtered = reference_labels
        n_excluded = 0
    
    metrics = {
        "reference_model": reference_name,
        "n_items_compared": int(len(detected_filtered)),
        "n_items_excluded": int(n_excluded),
        "ari": float(adjusted_rand_score(reference_filtered, detected_filtered)),
        "nmi": float(normalized_mutual_info_score(reference_filtered, detected_filtered)),
        "n_detected_clusters": int(len(np.unique(detected_filtered))),
        "n_reference_clusters": int(len(np.unique(reference_filtered))),
    }
    
    return metrics, valid_mask


def create_confusion_matrix_plot(detected_labels: np.ndarray,
                                 reference_labels: np.ndarray,
                                 detected_names: List[str],
                                 reference_names: List[str],
                                 reference_model_name: str,
                                 output_path: Path):
    """Create confusion matrix visualization."""
    # Create confusion matrix
    cm = confusion_matrix(reference_labels, detected_labels)
    
    # Normalize by rows (reference clusters)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=detected_names, yticklabels=reference_names,
                ax=ax1, cbar_kws={'label': 'Item Count'})
    ax1.set_xlabel('Detected Clusters', fontsize=12)
    ax1.set_ylabel(f'{reference_model_name} Factors', fontsize=12)
    ax1.set_title(f'Item Overlap (Raw Counts)\n{reference_model_name} vs Detected', fontsize=14)
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1,
                xticklabels=detected_names, yticklabels=reference_names,
                ax=ax2, cbar_kws={'label': 'Proportion'})
    ax2.set_xlabel('Detected Clusters', fontsize=12)
    ax2.set_ylabel(f'{reference_model_name} Factors', fontsize=12)
    ax2.set_title(f'Item Overlap (Row-Normalized)\n{reference_model_name} vs Detected', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_optimal_mapping(detected_labels: np.ndarray, 
                           reference_labels: np.ndarray,
                           n_detected: int,
                           n_reference: int) -> Dict[int, int]:
    """
    Compute optimal mapping from reference clusters to detected clusters using Hungarian algorithm.
    
    Args:
        detected_labels: Array of detected cluster labels
        reference_labels: Array of reference cluster labels
        n_detected: Number of detected clusters
        n_reference: Number of reference clusters
        
    Returns:
        mapping: dict {reference_cluster_id: detected_cluster_id}
    """
    # Create contingency matrix (reference × detected)
    contingency = np.zeros((n_reference, n_detected))
    
    for ref_c, det_c in zip(reference_labels, detected_labels):
        if ref_c >= 0 and det_c >= 0:  # Exclude noise points
            contingency[int(ref_c), int(det_c)] += 1
    
    # Use Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(-contingency)
    
    mapping = {int(ref_c): int(det_c) for ref_c, det_c in zip(row_ind, col_ind)}
    
    return mapping


def plot_sankey_diagram(detected_labels: np.ndarray,
                       reference_labels: np.ndarray,
                       detected_names: List[str],
                       reference_names: List[str],
                       optimal_mapping: Dict[int, int],
                       reference_model_name: str,
                       output_path: Path):
    """
    Create a Sankey-style visualization showing cluster flow from reference to detected.
    Uses matplotlib for a basic alluvial-style plot.
    
    Args:
        detected_labels: Array of detected cluster labels
        reference_labels: Array of reference cluster labels
        detected_names: List of detected cluster names
        reference_names: List of reference cluster names
        optimal_mapping: Dict mapping reference cluster IDs to detected cluster IDs
        reference_model_name: Name of reference model (for title)
        output_path: Path to save the plot
    """
    # Filter to valid pairs only
    valid_mask = (detected_labels >= 0) & (reference_labels >= 0)
    detected_valid = detected_labels[valid_mask]
    reference_valid = reference_labels[valid_mask]
    
    if len(detected_valid) == 0:
        print(f"      Warning: No valid pairs for {reference_model_name} Sankey diagram")
        return
    
    # Count flows between clusters
    flow_counts = {}
    for ref_c, det_c in zip(reference_valid, detected_valid):
        key = (int(ref_c), int(det_c))
        flow_counts[key] = flow_counts.get(key, 0) + 1
    
    # Create flow diagram
    fig, ax = plt.subplots(figsize=(12, 8))
    
    n_reference = len(reference_names)
    n_detected = len(detected_names)
    
    # Position clusters vertically
    ref_y = np.linspace(0, 1, n_reference)
    det_y = np.linspace(0, 1, n_detected)
    
    # Draw connections
    cmap = plt.cm.tab10
    for (ref_c, det_c), count in flow_counts.items():
        # Determine if this is the optimal mapping
        is_optimal = optimal_mapping.get(ref_c) == det_c
        alpha = 0.7 if is_optimal else 0.2
        color = cmap(ref_c % 10)
        
        # Draw connecting line
        ax.plot([0, 1], [ref_y[ref_c], det_y[det_c]], 
                linewidth=count*2, alpha=alpha, color=color, solid_capstyle='round')
    
    # Draw cluster nodes (reference on left)
    for i in range(n_reference):
        n_items = (reference_valid == i).sum()
        ax.scatter([0], [ref_y[i]], s=n_items*20, c=[cmap(i % 10)], 
                  edgecolors='black', linewidths=2, zorder=10)
        label = reference_names[i] if i < len(reference_names) else f'R{i}'
        ax.text(-0.05, ref_y[i], f'{label}\n(n={n_items})', 
               ha='right', va='center', fontsize=10, weight='bold')
    
    # Draw cluster nodes (detected on right)
    for i in range(n_detected):
        n_items = (detected_valid == i).sum()
        ax.scatter([1], [det_y[i]], s=n_items*20, c=[cmap(i % 10)], 
                  edgecolors='black', linewidths=2, zorder=10)
        label = detected_names[i] if i < len(detected_names) else f'D{i}'
        ax.text(1.05, det_y[i], f'{label}\n(n={n_items})', 
               ha='left', va='center', fontsize=10, weight='bold')
    
    ax.set_xlim([-0.2, 1.2])
    ax.set_ylim([-0.1, 1.1])
    ax.axis('off')
    ax.set_title(f'Factor Flow: {reference_model_name} → Detected (thick lines = optimal mapping)', 
                fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_sankey_reference_to_reference(theoretical_labels: np.ndarray,
                                       empirical_labels: np.ndarray,
                                       theoretical_names: List[str],
                                       empirical_names: List[str],
                                       output_path: Path):
    """
    Create a Sankey-style visualization comparing theoretical and empirical reference models.
    Shows how items flow between theoretical and empirical factor assignments.
    
    Args:
        theoretical_labels: Array of theoretical model labels
        empirical_labels: Array of empirical model labels
        theoretical_names: List of theoretical factor names
        empirical_names: List of empirical factor names
        output_path: Path to save the plot
    """
    # Filter to items assigned in both models
    valid_mask = (theoretical_labels >= 0) & (empirical_labels >= 0)
    theoretical_valid = theoretical_labels[valid_mask]
    empirical_valid = empirical_labels[valid_mask]
    
    if len(theoretical_valid) == 0:
        print(f"      Warning: No items assigned in both theoretical and empirical models")
        return
    
    # Compute optimal mapping from theoretical to empirical
    n_theoretical = len(theoretical_names)
    n_empirical = len(empirical_names)
    
    contingency = np.zeros((n_theoretical, n_empirical))
    for theo_c, emp_c in zip(theoretical_valid, empirical_valid):
        contingency[int(theo_c), int(emp_c)] += 1
    
    row_ind, col_ind = linear_sum_assignment(-contingency)
    optimal_mapping = {int(theo_c): int(emp_c) for theo_c, emp_c in zip(row_ind, col_ind)}
    
    # Count flows between clusters
    flow_counts = {}
    for theo_c, emp_c in zip(theoretical_valid, empirical_valid):
        key = (int(theo_c), int(emp_c))
        flow_counts[key] = flow_counts.get(key, 0) + 1
    
    # Create flow diagram
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Position clusters vertically
    theo_y = np.linspace(0, 1, n_theoretical)
    emp_y = np.linspace(0, 1, n_empirical)
    
    # Draw connections
    cmap = plt.cm.tab10
    for (theo_c, emp_c), count in flow_counts.items():
        # Determine if this is the optimal mapping
        is_optimal = optimal_mapping.get(theo_c) == emp_c
        alpha = 0.7 if is_optimal else 0.2
        color = cmap(theo_c % 10)
        
        # Draw connecting line
        ax.plot([0, 1], [theo_y[theo_c], emp_y[emp_c]], 
                linewidth=count*2, alpha=alpha, color=color, solid_capstyle='round')
    
    # Draw cluster nodes (theoretical on left)
    for i in range(n_theoretical):
        n_items = (theoretical_valid == i).sum()
        ax.scatter([0], [theo_y[i]], s=n_items*20, c=[cmap(i % 10)], 
                  edgecolors='black', linewidths=2, zorder=10)
        label = theoretical_names[i] if i < len(theoretical_names) else f'T{i}'
        ax.text(-0.05, theo_y[i], f'{label}\n(n={n_items})', 
               ha='right', va='center', fontsize=10, weight='bold')
    
    # Draw cluster nodes (empirical on right)
    for i in range(n_empirical):
        n_items = (empirical_valid == i).sum()
        ax.scatter([1], [emp_y[i]], s=n_items*20, c=[cmap(i % 10)], 
                  edgecolors='black', linewidths=2, zorder=10)
        label = empirical_names[i] if i < len(empirical_names) else f'E{i}'
        ax.text(1.05, emp_y[i], f'{label}\n(n={n_items})', 
               ha='left', va='center', fontsize=10, weight='bold')
    
    ax.set_xlim([-0.2, 1.2])
    ax.set_ylim([-0.1, 1.1])
    ax.axis('off')
    ax.set_title('Factor Flow: Theoretical → Empirical (thick lines = optimal mapping)', 
                fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_item_assignment_comparison(detected_df: pd.DataFrame,
                                      reference_dict: Dict[str, List[str]],
                                      reference_name: str,
                                      output_path: Path,
                                      cluster_label_dict: Dict[int, str] = None):
    """Create detailed item-by-item assignment comparison.
    
    Args:
        detected_df: DataFrame with item_id and cluster columns
        reference_dict: Dict mapping factor names to item lists
        reference_name: Name of reference model (for context)
        output_path: Path to save CSV output
        cluster_label_dict: Dict mapping cluster IDs to factor names
    """
    # Build reference assignment dictionary
    item_to_reference = {}
    for factor, items in reference_dict.items():
        for item in items:
            if item not in item_to_reference:
                item_to_reference[item] = factor
            else:
                # Handle items assigned to multiple factors
                item_to_reference[item] = f"{item_to_reference[item]},{factor}"
    
    # Create comparison dataframe
    comparison = detected_df.copy()
    comparison['reference_factor'] = comparison['item_id'].map(item_to_reference)
    comparison['reference_factor'] = comparison['reference_factor'].fillna('unassigned')
    
    # Add detected cluster factor mapping
    if cluster_label_dict:
        comparison['detected_factor'] = comparison['cluster'].map(cluster_label_dict)
        comparison['detected_factor'] = comparison['detected_factor'].fillna('unassigned')
    else:
        comparison['detected_factor'] = comparison['cluster'].astype(str)
    
    # Compute agreement: the detected factor matches ANY of the reference
    # factors listed for the item ("any-factor" rule; multi-factor items are
    # comma-separated).  Items without a reference factor never agree.
    def _canon(name: str) -> str:
        key = str(name).strip().lower().replace("-", " ").replace("_", " ")
        key = "_".join(key.split())
        # the post-hoc cluster label reads "higher order processes"
        return {"higher_order_processes": "higher_order_processing"}.get(key, key)

    def check_agreement(row):
        if row['reference_factor'] == 'unassigned':
            return False
        reference_factors = [_canon(f) for f in str(row['reference_factor']).split(',')]
        return _canon(row['detected_factor']) in reference_factors
    
    comparison['agrees'] = comparison.apply(check_agreement, axis=1)
    
    # Save to CSV
    comparison.to_csv(output_path, index=False)
    
    return comparison


def generate_comparison_report(metrics_theoretical: Dict,
                               metrics_empirical: Dict,
                               output_path: Path):
    """Generate markdown report of model comparisons."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Measurement Model Comparison Report\n\n")
        f.write("## Overview\n\n")
        f.write("Comparison of data-driven clustering solution to reference measurement models.\n\n")
        
        f.write("## Metrics Summary\n\n")
        f.write("| Metric | Theoretical Model | Empirical Model |\n")
        f.write("|--------|------------------|----------------|\n")
        f.write(f"| **Adjusted Rand Index (ARI)** | {metrics_theoretical['ari']:.3f} | {metrics_empirical['ari']:.3f} |\n")
        f.write(f"| **Normalized Mutual Info (NMI)** | {metrics_theoretical['nmi']:.3f} | {metrics_empirical['nmi']:.3f} |\n")
        f.write(f"| **Items Compared** | {metrics_theoretical['n_items_compared']} | {metrics_empirical['n_items_compared']} |\n")
        f.write(f"| **Detected Clusters** | {metrics_theoretical['n_detected_clusters']} | {metrics_empirical['n_detected_clusters']} |\n")
        f.write(f"| **Reference Factors** | {metrics_theoretical['n_reference_clusters']} | {metrics_empirical['n_reference_clusters']} |\n\n")
        
        f.write("## Interpretation\n\n")
        f.write("**Adjusted Rand Index (ARI)**:\n")
        f.write("- Range: -1 to 1 (1 = perfect agreement, 0 = random)\n")
        f.write("- Values > 0.5 indicate strong agreement\n")
        f.write("- Values > 0.7 indicate very strong agreement\n\n")
        
        f.write("**Normalized Mutual Information (NMI)**:\n")
        f.write("- Range: 0 to 1 (1 = perfect agreement)\n")
        f.write("- Values > 0.5 indicate moderate-to-strong shared information\n\n")
        
        # Determine which model aligns better
        if metrics_theoretical['ari'] > metrics_empirical['ari']:
            f.write("**Conclusion**: The detected clustering shows stronger alignment with the **theoretical model**.\n")
        elif metrics_empirical['ari'] > metrics_theoretical['ari']:
            f.write("**Conclusion**: The detected clustering shows stronger alignment with the **empirical model**.\n")
        else:
            f.write("**Conclusion**: The detected clustering shows equal alignment with both reference models.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare detected clustering to reference measurement models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["all-mpnet-base-v2", "all-roberta-large-v1"],
        help="Model name (auto-sets paths)"
    )
    parser.add_argument(
        "--cluster-file",
        type=str,
        default=None,
        help="Path to cluster_assignments.csv from Step 03 (default: data/outputs/experiments/<model>/03_clustering/)"
    )
    parser.add_argument(
        "--cluster-labels-file",
        type=str,
        default=None,
        help="Path to cluster_assignments.csv from Step 04 (default: data/outputs/experiments/<model>/04_labeling/)"
    )
    parser.add_argument(
        "--reference-models",
        type=str,
        default=None,
        help="Path to reference models YAML file (default: config/reference_models_b4.yaml)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for comparison results (default: data/outputs/experiments/<model>/05_model_comparison/)"
    )
    
    args = parser.parse_args()
    
    # Auto-configure paths
    root = Path(__file__).resolve().parents[2]
    
    # Auto-set model slug from cluster file if not provided
    if args.model:
        model_slug = args.model.replace("-", "_")
    elif args.cluster_file:
        # Try to infer model from cluster file path
        cluster_path = Path(args.cluster_file)
        if "all_mpnet_base_v2" in str(cluster_path) or "all-mpnet-base-v2" in str(cluster_path):
            model_slug = "all_mpnet_base_v2"
            args.model = "all-mpnet-base-v2"
        elif "all_roberta_large_v1" in str(cluster_path) or "all-roberta-large-v1" in str(cluster_path):
            model_slug = "all_roberta_large_v1"
            args.model = "all-roberta-large-v1"
        else:
            model_slug = None
    else:
        model_slug = None
    
    # Set defaults based on model
    if model_slug:
        if args.cluster_file is None:
            args.cluster_file = str(root / "data" / "outputs" / "experiments" / f"{model_slug}_analysis" / "03_clustering" / "cluster_assignments.csv")
        if args.cluster_labels_file is None:
            args.cluster_labels_file = str(root / "data" / "outputs" / "experiments" / f"{model_slug}_analysis" / "04_labeling" / "cluster_concept_labels.csv")
        if args.output_dir is None:
            args.output_dir = str(root / "data" / "outputs" / "experiments" / f"{model_slug}_analysis" / "05_model_comparison")
    
    # Validate that we have required arguments
    if args.cluster_file is None:
        parser.error(
            "Either --model or --cluster-file must be provided. "
            "Use --model all-mpnet-base-v2 (or all-roberta-large-v1) to auto-detect paths, "
            "or --cluster-file to specify the path directly."
        )
    
    if args.output_dir is None:
        # Try to infer from cluster_file if it wasn't already set
        cluster_path = Path(args.cluster_file)
        if cluster_path.parent.name == "03_clustering":
            # Extract the experiments/<model_slug>_analysis directory
            analysis_dir = cluster_path.parent.parent
            args.output_dir = str(analysis_dir / "05_model_comparison")
        else:
            parser.error(
                "Could not infer output directory. "
                "Please provide --output-dir explicitly."
            )
    
    if args.reference_models is None:
        args.reference_models = str(root / "config" / "reference_models_b4.yaml")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("STEP 5: MEASUREMENT MODEL COMPARISON")
    print(f"{'='*70}\n")
    print(f"Cluster assignments: {args.cluster_file}")
    print(f"Cluster labels: {args.cluster_labels_file}")
    print(f"Reference models: {args.reference_models}")
    print(f"Output directory: {output_dir}\n")
    
    # Load data
    print("[1/5] Loading cluster assignments...")
    detected_df = pd.read_csv(args.cluster_file)
    all_items = detected_df['item_id'].tolist()
    detected_labels = detected_df['cluster'].values
    print(f"      Loaded {len(all_items)} items with {len(np.unique(detected_labels))} clusters")
    
    # Load cluster labels from step 04
    print("\n[2/5] Loading cluster labels...")
    cluster_label_names, cluster_label_dict = load_cluster_labels(Path(args.cluster_labels_file))
    print(f"      Cluster labels: {cluster_label_names}")
    ref_models = load_reference_models(Path(args.reference_models))
    print(f"      Loaded reference models: {list(ref_models.keys())}")
    
    # Convert reference models to label arrays
    theoretical_dict = {k: v for k, v in ref_models['theoretical'].items() if isinstance(v, list)}
    empirical_dict = {k: v for k, v in ref_models['empirical'].items() if isinstance(v, list)}
    
    theoretical_labels = convert_factor_dict_to_labels(theoretical_dict, all_items)
    empirical_labels = convert_factor_dict_to_labels(empirical_dict, all_items)
    
    # Build multi-factor mappings (for handling items in multiple factors)
    theoretical_item_to_factors = convert_factor_dict_to_item_factor_map(theoretical_dict, all_items)
    empirical_item_to_factors = convert_factor_dict_to_item_factor_map(empirical_dict, all_items)
    
    # Expand labels for items that belong to multiple factors
    detected_expanded_th, theoretical_expanded, _ = expand_labels_for_multi_factor(
        all_items, detected_labels, theoretical_item_to_factors
    )
    detected_expanded_em, empirical_expanded, _ = expand_labels_for_multi_factor(
        all_items, detected_labels, empirical_item_to_factors
    )
    
    print(f"      Theoretical model: {len(theoretical_dict)} factors")
    print(f"      Empirical model: {len(empirical_dict)} factors")
    print(f"      Note: Items with multi-factor assignments are counted in all their factors")
    
    # Compute alignment metrics
    print("\n[3/5] Computing alignment metrics...")
    metrics_theoretical, mask_theoretical = compute_alignment_metrics(
        detected_expanded_th, theoretical_expanded, "Theoretical"
    )
    metrics_empirical, mask_empirical = compute_alignment_metrics(
        detected_expanded_em, empirical_expanded, "Empirical"
    )
    
    print(f"      Theoretical ARI: {metrics_theoretical['ari']:.3f}")
    print(f"      Empirical ARI: {metrics_empirical['ari']:.3f}")
    
    # Save metrics
    metrics_summary = {
        "theoretical": metrics_theoretical,
        "empirical": metrics_empirical,
    }
    with open(output_dir / "comparison_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"      Metrics saved to: {output_dir / 'comparison_metrics.json'}")
    
    # Create visualizations
    print("\n[4/5] Creating visualizations...")
    
    # Compute optimal mappings for Sankey diagrams
    n_detected = len(np.unique(detected_labels[detected_labels >= 0]))
    n_theoretical = len(theoretical_dict)
    n_empirical = len(empirical_dict)
    
    mapping_theoretical = compute_optimal_mapping(
        detected_expanded_th[mask_theoretical],
        theoretical_expanded[mask_theoretical],
        n_detected,
        n_theoretical
    )
    
    mapping_empirical = compute_optimal_mapping(
        detected_expanded_em[mask_empirical],
        empirical_expanded[mask_empirical],
        n_detected,
        n_empirical
    )
    
    # Confusion matrices
    create_confusion_matrix_plot(
        detected_expanded_th[mask_theoretical],
        theoretical_expanded[mask_theoretical],
        cluster_label_names,
        list(theoretical_dict.keys()),
        "Theoretical",
        output_dir / "confusion_matrix_theoretical.png"
    )
    
    create_confusion_matrix_plot(
        detected_expanded_em[mask_empirical],
        empirical_expanded[mask_empirical],
        cluster_label_names,
        list(empirical_dict.keys()),
        "Empirical",
        output_dir / "confusion_matrix_empirical.png"
    )
    print(f"      Confusion matrices saved to: {output_dir}")
    
    # Sankey diagrams
    plot_sankey_diagram(
        detected_expanded_th[mask_theoretical],
        theoretical_expanded[mask_theoretical],
        cluster_label_names,
        list(theoretical_dict.keys()),
        mapping_theoretical,
        "Theoretical",
        output_dir / "sankey_theoretical_to_detected.png"
    )
    
    plot_sankey_diagram(
        detected_expanded_em[mask_empirical],
        empirical_expanded[mask_empirical],
        cluster_label_names,
        list(empirical_dict.keys()),
        mapping_empirical,
        "Empirical",
        output_dir / "sankey_empirical_to_detected.png"
    )
    print(f"      Sankey diagrams saved to: {output_dir}")
    
    # Sankey diagram comparing theoretical to empirical
    # Need to use the original (non-expanded) labels for direct comparison
    plot_sankey_reference_to_reference(
        theoretical_labels,
        empirical_labels,
        list(theoretical_dict.keys()),
        list(empirical_dict.keys()),
        output_dir / "sankey_theoretical_to_empirical.png"
    )
    print(f"      Theoretical-Empirical comparison diagram saved")
    
    # Item-level comparisons
    create_item_assignment_comparison(
        detected_df, theoretical_dict, "Theoretical",
        output_dir / "item_comparison_theoretical.csv",
        cluster_label_dict
    )
    create_item_assignment_comparison(
        detected_df, empirical_dict, "Empirical",
        output_dir / "item_comparison_empirical.csv",
        cluster_label_dict
    )
    print(f"      Item-level comparisons saved to: {output_dir}")
    
    # Generate report
    print("\n[5/5] Generating comparison report...")
    generate_comparison_report(
        metrics_theoretical,
        metrics_empirical,
        output_dir / "comparison_report.md"
    )
    print(f"      Report saved to: {output_dir / 'comparison_report.md'}")
    
    print(f"\n{'='*70}")
    print("✓ MODEL COMPARISON COMPLETE!")
    print(f"{'='*70}\n")
    print(f"Results saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
