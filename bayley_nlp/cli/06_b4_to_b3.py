
# --- UTF-8 stdout/stderr: this script prints Unicode (arrows, checkmarks)
# which crashes under Windows' default cp1252 encoding; force UTF-8. ---
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")
del _stream

#!/usr/bin/env python
"""
Step 06: Pair B4 and B3 items by embedding similarity and visualize with MDS.

This script:
1) Loads B4 and B3 items
2) Embeds both item sets with the same model
3) Greedily pairs items (1 B4 with 1 B3, exclusive)
4) Projects all items to 2D using MDS and plots pair connections
5) Assigns B3 items to the same clusters as their paired B4 items
6) Saves the new model specification to model_specs.yaml

Usage:
    python 06_b4_to_b3.py --model all-mpnet-base-v2
    python 06_b4_to_b3.py --model all-roberta-large-v1 --annotate
"""

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics.pairwise import cosine_similarity

from bayley_nlp.core import parse_items, load_model, embed_items, ensure_dir
from bayley_nlp.core.dimensionality_reduction import get_reducer


def parse_args():
    ap = argparse.ArgumentParser(
        description="Pair B4 and B3 items by embedding similarity and visualize with MDS",
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
        "--items-file-b4",
        type=str,
        default=None,
        help="B4 items file (default: data/raw/items_description/b4_cog_items.txt)",
    )
    ap.add_argument(
        "--items-file-b3",
        type=str,
        default=None,
        help="B3 items file (default: data/raw/items_description/b3_cog_items.txt)",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/outputs/experiments/<model>/06_b4_b3_pairing)",
    )
    ap.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate points with item IDs in the 2D plot",
    )
    ap.add_argument(
        "--figure-size",
        type=str,
        default="10,8",
        help="Figure size as 'width,height' in inches",
    )
    return ap.parse_args()


def greedy_pairing(sim_matrix: np.ndarray):
    """
    Greedily pair items across two sets using a similarity matrix.

    Returns:
        pairs: list of (i_b4, i_b3, similarity)
        used_b4: set of paired b4 indices
        used_b3: set of paired b3 indices
    """
    n_b4, n_b3 = sim_matrix.shape
    flat_order = np.argsort(sim_matrix.ravel())[::-1]
    pairs = []
    used_b4 = set()
    used_b3 = set()

    for flat_idx in flat_order:
        i = flat_idx // n_b3
        j = flat_idx % n_b3
        if i in used_b4 or j in used_b3:
            continue
        pairs.append((i, j, float(sim_matrix[i, j])))
        used_b4.add(i)
        used_b3.add(j)
        if len(used_b4) == n_b4 or len(used_b3) == n_b3:
            break

    return pairs, used_b4, used_b3


def save_model_spec(b3_items, b3_clusters_df, paired_df, model, root, output_dir):
    """
    Create and save a model specification for B3 items based on B4 clustering.
    Renumbers factors from F0-based to F1-based and includes concept labels.
    
    Returns:
        model_name (str): The key used in model_specs.yaml
    """
    # Load existing model specs to get factor names and indices from B4 model
    config_dir = root / "config"
    model_specs_file = config_dir / "model_specs.yaml"
    
    with open(model_specs_file, 'r') as f:
        existing_specs = yaml.safe_load(f)
    
    # Find the corresponding B4 consensus model
    model_slug = model.replace("/", "_").replace("-", "_")
    b4_model_key = None
    for key in existing_specs.keys():
        if model_slug in key and "consensus" in key:
            b4_model_key = key
            break
    
    if b4_model_key is None:
        raise ValueError(f"Could not find consensus model for {model} in model_specs.yaml")
    
    b4_model = existing_specs[b4_model_key]
    n_factors = b4_model['n_factors']
    
    # Get unique clusters from the data (will be 0 to n_factors-1)
    unique_clusters = sorted([c for c in b3_clusters_df['cluster'].unique() if c != -1])
    
    # Build factor information for B3 - renumber to F1, F2, etc. (not F0, F1, ...)
    factor_indices = {f'F{i+1}': [] for i in unique_clusters}
    factor_items = {f'F{i+1}': [] for i in unique_clusters}
    factor_sizes = {f'F{i+1}': 0 for i in unique_clusters}
    factor_stability_scores = {f'F{i+1}': 0.0 for i in unique_clusters}
    
    # Collect data by cluster
    cluster_stability = {f'F{i+1}': [] for i in unique_clusters}
    
    for idx, row in b3_clusters_df.iterrows():
        if row['cluster'] == -1:
            continue  # Skip noise points
        
        cluster_num = int(row['cluster'])
        factor_key = f'F{cluster_num+1}'  # Renumber: 0->F1, 1->F2, etc.
        item_id = row['item_id']
        probability = row['probability']
        
        factor_indices[factor_key].append(idx)
        factor_items[factor_key].append(item_id)
        factor_sizes[factor_key] += 1
        cluster_stability[factor_key].append(probability)
    
    # Calculate stability scores per factor
    for factor_key in factor_stability_scores.keys():
        if cluster_stability[factor_key]:
            factor_stability_scores[factor_key] = float(np.mean(cluster_stability[factor_key]))
        else:
            factor_stability_scores[factor_key] = 0.0
    
    # Calculate overall stability metrics
    all_probabilities = b3_clusters_df[b3_clusters_df['cluster'] != -1]['probability'].values
    mean_item_stability = float(np.mean(all_probabilities)) if len(all_probabilities) > 0 else 0.0
    median_item_stability = float(np.median(all_probabilities)) if len(all_probabilities) > 0 else 0.0
    min_item_stability = float(np.min(all_probabilities)) if len(all_probabilities) > 0 else 0.0
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Load concept labels from 04_labeling if available
    concept_labels_file = output_dir.parent / "04_labeling" / "cluster_concept_labels.csv"
    
    factor_concepts = {}
    if concept_labels_file.exists():
        concept_df = pd.read_csv(concept_labels_file)
        # Map cluster_id (0-based) to concept labels
        for _, row in concept_df.iterrows():
            cluster_id = int(row['cluster_id'])
            factor_key = f'F{cluster_id+1}'  # Renumber to match factor naming
            factor_concepts[factor_key] = {
                'concept': row['top_concept'],
                'similarity': float(row['top_concept_similarity']),
            }
    
    # Create model specification
    model_spec = {
        'source': 'b4_b3_pairing',
        'model': model,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'n_factors': n_factors,
        'n_items': len(b3_items),
        'method': f'Items paired from {b4_model_key} and assigned to same factors',
        'pairing_method': 'Greedy pairing by embedding similarity',
        'n_pairs': len(paired_df),
        'mean_pairing_similarity': float(paired_df['similarity'].mean()),
        'factor_indices': factor_indices,
        'factor_items': factor_items,
        'factor_sizes': factor_sizes,
        'factor_stability_scores': factor_stability_scores,
        'factor_concepts': factor_concepts if factor_concepts else None,
        'stability_metrics': {
            'mean_item_stability': mean_item_stability,
            'median_item_stability': median_item_stability,
            'min_item_stability': min_item_stability,
        },
        'timestamp': timestamp,
        'description': (
            f"B3 items paired with {b4_model_key} items using embedding similarity. "
            f"Each B3 item is assigned to the same factor as its paired B4 item. "
            f"Mean pairing similarity: {paired_df['similarity'].mean():.4f}. "
            f"Total paired items: {len(paired_df)} out of {len(b3_items)}."
        ),
    }
    
    # Remove factor_concepts if empty to keep YAML clean
    if not factor_concepts:
        del model_spec['factor_concepts']
    
    # Generate model name
    model_name = f"{model_slug}_b3_from_b4_pairing"
    
    # Load existing specs and add new one
    with open(model_specs_file, 'r') as f:
        all_specs = yaml.safe_load(f) or {}
    
    all_specs[model_name] = model_spec
    
    # Save back to file
    with open(model_specs_file, 'w') as f:
        yaml.dump(all_specs, f, default_flow_style=False, sort_keys=False)
    
    # Also save a standalone copy in the output directory
    standalone_spec = {model_name: model_spec}
    spec_file = output_dir / "b3_model_spec.yaml"
    with open(spec_file, 'w') as f:
        yaml.dump(standalone_spec, f, default_flow_style=False, sort_keys=False)
    
    return model_name



def main():
    args = parse_args()

    root = Path(__file__).resolve().parents[2]
    model_slug = args.model.replace("/", "_").replace("-", "_")

    def resolve_items_path(arg_value: str | None, default_name: str) -> Path:
        base_dir = root / "data" / "raw" / "items_description"
        if arg_value is None:
            return base_dir / default_name
        candidate = Path(arg_value)
        if candidate.is_absolute():
            return candidate
        root_candidate = root / candidate
        if root_candidate.exists():
            return root_candidate
        return base_dir / candidate

    items_file_b4 = resolve_items_path(args.items_file_b4, "b4_cog_items.txt")
    items_file_b3 = resolve_items_path(args.items_file_b3, "b3_cog_items.txt")

    if args.output_dir is None:
        output_dir = root / "data" / "outputs" / "experiments" / f"{model_slug}_analysis" / "06_b4_b3_pairing"
    else:
        output_dir = Path(args.output_dir)
        if output_dir.name != "06_b4_b3_pairing":
            output_dir = output_dir / "06_b4_b3_pairing"

    ensure_dir(str(output_dir))

    print(f"\n{'='*70}")
    print("STEP 06: B4 <-> B3 GREEDY PAIRING & CLUSTER ASSIGNMENT")
    print(f"{'='*70}\n")
    print(f"Model: {args.model}")
    print(f"B4 items: {items_file_b4}")
    print(f"B3 items: {items_file_b3}")
    print(f"Output directory: {output_dir}\n")

    print("[1/7] Parsing items...")
    b4_items = parse_items(str(items_file_b4))
    b3_items = parse_items(str(items_file_b3))
    print(f"      Parsed {len(b4_items)} B4 items")
    print(f"      Parsed {len(b3_items)} B3 items")

    print("\n[2/7] Generating embeddings...")
    model = load_model(args.model)
    b4_emb = embed_items(b4_items, text_column="item_text", model=model)
    b3_emb = embed_items(b3_items, text_column="item_text", model=model)
    print(f"      B4 embeddings: {b4_emb.shape}")
    print(f"      B3 embeddings: {b3_emb.shape}")

    print("\n[3/7] Computing greedy pairing...")
    sim_matrix = cosine_similarity(b4_emb, b3_emb)
    pairs, used_b4, used_b3 = greedy_pairing(sim_matrix)
    print(f"      Paired {len(pairs)} items (min of {len(b4_items)} and {len(b3_items)})")

    paired_rows = []
    for i, j, sim in pairs:
        paired_rows.append(
            {
                "b4_item_id": b4_items.loc[i, "item_id"],
                "b4_item_title": b4_items.loc[i, "item_title"],
                "b3_item_id": b3_items.loc[j, "item_id"],
                "b3_item_title": b3_items.loc[j, "item_title"],
                "similarity": sim,
            }
        )

    paired_df = pd.DataFrame(paired_rows)
    paired_csv = output_dir / "b4_b3_greedy_pairs.csv"
    paired_df.to_csv(paired_csv, index=False)
    print(f"      Saved pairs: {paired_csv}")

    # Save unpaired items if any
    unpaired_b4 = sorted(set(range(len(b4_items))) - used_b4)
    unpaired_b3 = sorted(set(range(len(b3_items))) - used_b3)

    if unpaired_b4:
        b4_unpaired_df = b4_items.loc[unpaired_b4, ["item_id", "item_title"]].copy()
        b4_unpaired_df.to_csv(output_dir / "b4_unpaired_items.csv", index=False)
    if unpaired_b3:
        b3_unpaired_df = b3_items.loc[unpaired_b3, ["item_id", "item_title"]].copy()
        b3_unpaired_df.to_csv(output_dir / "b3_unpaired_items.csv", index=False)

    print("\n[4/7] Loading B4 cluster assignments...")
    cluster_file = output_dir.parent / "03_clustering" / "cluster_assignments.csv"
    
    if not cluster_file.exists():
        raise FileNotFoundError(f"Cluster assignments file not found: {cluster_file}\nRun clustering first: python bayley_nlp/cli/03_cluster_1000_kmean.py")
    
    b4_clusters = pd.read_csv(cluster_file)
    print(f"      Loaded cluster assignments for {len(b4_clusters)} B4 items")

    print("\n[5/7] Assigning B3 items to B4 cluster structure...")
    # Create mapping from item_id to cluster for B4
    b4_cluster_map = dict(zip(b4_clusters['item_id'], b4_clusters['cluster']))
    b4_prob_map = dict(zip(b4_clusters['item_id'], b4_clusters['probability']))
    
    # Assign B3 items to same clusters as their paired B4 items
    b3_assignments = []
    for _, row in paired_df.iterrows():
        b4_id = row['b4_item_id']
        b3_id = row['b3_item_id']
        cluster = b4_cluster_map[b4_id]
        probability = b4_prob_map[b4_id]
        b3_assignments.append({
            'item_id': b3_id,
            'cluster': cluster,
            'probability': probability,
        })
    
    # Add unpaired B3 items as noise (-1)
    for _, row in b3_unpaired_df.iterrows():
        b3_assignments.append({
            'item_id': row['item_id'],
            'cluster': -1,
            'probability': 0.0,
        })
    
    b3_clusters_df = pd.DataFrame(b3_assignments)
    b3_cluster_csv = output_dir / "b3_cluster_assignments.csv"
    b3_clusters_df.to_csv(b3_cluster_csv, index=False)
    print(f"      Saved B3 cluster assignments: {b3_cluster_csv}")

    print("\n[6/7] Reducing to 2D with MDS...")
    all_emb = np.vstack([b4_emb, b3_emb])
    reducer = get_reducer("mds", n_components=2, random_state=42, verbose=True)
    coords_2d = reducer.fit_transform(all_emb)

    coords_df = pd.DataFrame(
        {
            "item_id": pd.concat([b4_items["item_id"], b3_items["item_id"]], ignore_index=True),
            "test": ["B4"] * len(b4_items) + ["B3"] * len(b3_items),
            "x": coords_2d[:, 0],
            "y": coords_2d[:, 1],
        }
    )
    coords_csv = output_dir / "b4_b3_mds_coords.csv"
    coords_df.to_csv(coords_csv, index=False)
    print(f"      Saved coordinates: {coords_csv}")

    print("\n[7/7] Creating visualization and saving model spec...")
    try:
        fig_w, fig_h = [float(x.strip()) for x in args.figure_size.split(",")]
    except ValueError:
        fig_w, fig_h = 10.0, 8.0

    b4_coords = coords_2d[: len(b4_items)]
    b3_coords = coords_2d[len(b4_items) :]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.scatter(b4_coords[:, 0], b4_coords[:, 1], s=36, c="#1f77b4", label="B4", alpha=0.8)
    ax.scatter(b3_coords[:, 0], b3_coords[:, 1], s=36, c="#ff7f0e", label="B3", alpha=0.8)

    # Draw lines for paired items
    for i, j, _sim in pairs:
        x_vals = [b4_coords[i, 0], b3_coords[j, 0]]
        y_vals = [b4_coords[i, 1], b3_coords[j, 1]]
        ax.plot(x_vals, y_vals, color="gray", alpha=0.3, linewidth=0.8)

    if args.annotate:
        for idx, row in b4_items.iterrows():
            ax.text(b4_coords[idx, 0], b4_coords[idx, 1], row["item_id"], fontsize=7)
        for idx, row in b3_items.iterrows():
            ax.text(b3_coords[idx, 0], b3_coords[idx, 1], row["item_id"], fontsize=7)

    ax.set_title("B4-B3 Greedy Pairing (MDS 2D)")
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.2)

    plot_path = output_dir / "b4_b3_mds_pairs.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    print(f"      Saved plot: {plot_path}")
    
    # Generate and save model specification
    model_name = save_model_spec(
        b3_items=b3_items,
        b3_clusters_df=b3_clusters_df,
        paired_df=paired_df,
        model=args.model,
        root=root,
        output_dir=output_dir,
    )
    
    print(f"      Saved model spec: {model_name}")

    print(f"\n{'='*70}")
    print("STEP 06 COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
