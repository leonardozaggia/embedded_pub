"""
Cluster Centroid Labeling via Semantic Decoding

This script computes the centroid (mean embedding) for each cluster and decodes it
into interpretable text labels. The centroid represents the average semantic meaning
of all items that cluster together.

Methods:
1. Centroid-based: Average embeddings within each cluster
2. Nearest neighbors: Find items closest to the centroid
3. LLM summarization (optional): Use language model to generate labels from centroid neighbors

How to run:
    python bayley_nlp/cli/04_labeling.py --model all-mpnet-base-v2 --decode-concepts --visualize
    python bayley_nlp/cli/04_labeling.py --model all-roberta-large-v1 --decode-concepts --visualize --items data/processed/custom_items.csv 
"""


from __future__ import annotations

# --- UTF-8 stdout/stderr: this script prints Unicode (arrows, checkmarks)
# which crashes under Windows' default cp1252 encoding; force UTF-8. ---
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")
del _stream

import argparse
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from bayley_nlp.core.parsing import parse_items
from bayley_nlp.core.embeddings import load_model, embed_items


def compute_cluster_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> Dict[int, np.ndarray]:
    """
    Compute the centroid (mean embedding) for each cluster.
    
    Args:
        embeddings: Item embeddings (n_items, embedding_dim)
        labels: Cluster labels for each item
        
    Returns:
        Dictionary mapping cluster_id -> centroid vector
    """
    centroids = {}
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        if label == -1:  # Skip noise points if any
            continue
        mask = labels == label
        cluster_embeddings = embeddings[mask]
        # Compute mean and re-normalize
        centroid = cluster_embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids[int(label)] = centroid
    
    return centroids


def find_nearest_items(
    centroid: np.ndarray,
    embeddings: np.ndarray,
    item_ids: List[str],
    item_texts: List[str],
    top_k: int = 5
) -> pd.DataFrame:
    """
    Find the k items most similar to the centroid.
    
    Args:
        centroid: Cluster centroid vector
        embeddings: All item embeddings
        item_ids: Item identifiers
        item_texts: Item text descriptions
        top_k: Number of nearest items to return
        
    Returns:
        DataFrame with nearest items and their similarities
    """
    # Compute cosine similarities
    similarities = cosine_similarity(centroid.reshape(1, -1), embeddings)[0]
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Create result dataframe
    results = pd.DataFrame({
        'item_id': [item_ids[i] for i in top_indices],
        'item_text': [item_texts[i] for i in top_indices],
        'similarity': similarities[top_indices]
    })
    
    return results


def generate_cluster_labels(
    centroids: Dict[int, np.ndarray],
    embeddings: np.ndarray,
    items: pd.DataFrame,
    top_k: int = 5,
    text_column: str = "item_text"
) -> pd.DataFrame:
    """
    Generate interpretable labels for each cluster by finding nearest items.
    
    Args:
        centroids: Dictionary of cluster centroids
        embeddings: All item embeddings
        items: DataFrame with item metadata
        top_k: Number of representative items per cluster
        text_column: Column name with item text
        
    Returns:
        DataFrame with cluster labels and representative items
    """
    item_ids = items['item_id'].tolist()
    item_texts = items[text_column].tolist()
    
    cluster_info = []
    
    for cluster_id, centroid in sorted(centroids.items()):
        # Find nearest items to centroid
        nearest = find_nearest_items(
            centroid, embeddings, item_ids, item_texts, top_k
        )
        
        # Get items in this cluster
        cluster_mask = items.index.isin(
            items[items['item_id'].isin(nearest['item_id'])].index
        )
        
        cluster_info.append({
            'cluster_id': cluster_id,
            'n_items': int(np.sum(cluster_mask)),
            'representative_items': ', '.join(nearest['item_id'].head(3).tolist()),
            'top_similarity': float(nearest['similarity'].iloc[0]),
            'mean_similarity': float(nearest['similarity'].mean()),
            'nearest_items_df': nearest
        })
    
    return pd.DataFrame(cluster_info)


def decode_centroid_to_text(
    centroid: np.ndarray,
    model,
    candidate_texts: Optional[List[str]] = None,
    candidate_names: Optional[List[str]] = None,
    top_k: int = 10,
    use_names_only: bool = False
) -> List[Tuple[str, float]]:
    """
    Decode a centroid vector by finding most similar texts from a candidate set.
    
    This is useful when you want to generate a descriptive label by comparing
    the centroid to a set of concept descriptions or keywords.
    
    Args:
        centroid: Cluster centroid vector
        model: Embedding model (loaded)
        candidate_texts: Optional list of candidate concept descriptions (name: definition)
        candidate_names: Optional list of just construct names
        top_k: Number of top candidates to return
        use_names_only: If True, embed and compare using just names instead of full descriptions
        
    Returns:
        List of (text, similarity) tuples
    """
    if use_names_only and candidate_names is not None:
        # Use names only mode (ignore descriptions)
        texts_to_embed = candidate_names
        texts_for_display = candidate_names
    elif candidate_texts is not None:
        # Use full descriptions (name: definition)
        texts_to_embed = candidate_texts
        texts_for_display = candidate_texts
    else:
        # Default cognitive domain keywords
        texts_to_embed = [
            "flexibility/shift",
            "working memory",
            "attention",
            "higher order processing",
            "goal directed problem solving",
        ]
        texts_for_display = texts_to_embed
    
    # Embed candidate texts
    candidate_df = pd.DataFrame({'text': texts_to_embed})
    candidate_embeddings = embed_items(
        candidate_df, 
        text_column='text', 
        model=model
    )
    
    # Compute similarities
    similarities = cosine_similarity(
        centroid.reshape(1, -1), 
        candidate_embeddings
    )[0]
    
    # Sort by similarity
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = [
        (texts_for_display[i], float(similarities[i])) 
        for i in top_indices
    ]
    
    return results


def save_cluster_labels(
    cluster_labels: pd.DataFrame,
    outdir: str,
    include_details: bool = True
):
    """
    Save cluster labeling results to files.
    
    Args:
        cluster_labels: DataFrame with cluster information
        outdir: Output directory
        include_details: Whether to save detailed nearest items for each cluster
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Save summary
    summary_cols = [
        'cluster_id', 'n_items', 'representative_items', 
        'top_similarity', 'mean_similarity'
    ]
    cluster_labels[summary_cols].to_csv(
        os.path.join(outdir, 'cluster_labels_summary.csv'),
        index=False
    )
    
    # Save detailed nearest items for each cluster
    if include_details:
        for _, row in cluster_labels.iterrows():
            cluster_id = row['cluster_id']
            nearest_df = row['nearest_items_df']
            
            filename = f'cluster_{cluster_id}_nearest_items.csv'
            nearest_df.to_csv(
                os.path.join(outdir, filename),
                index=False
            )
    
    print(f"Cluster labels saved to: {outdir}")


def load_candidate_texts_from_yaml(yaml_path: str, section: str = "oxford", use_names_only: bool = False) -> Tuple[List[str], List[str]]:
    """
    Load candidate texts from the construct definition YAML file.
    
    Args:
        yaml_path: Path to construct_definition.yaml
        section: Which section to load ("oxford", "cognitive_psychology", or "average")
                 If "average", uses both oxford and cognitive_psychology and averages their embeddings later
        use_names_only: If True, return only construct names (ignore descriptions)
        
    Returns:
        Tuple of (full_texts, construct_names) where:
        - full_texts: List of "construct: definition" strings (for embedding)
        - construct_names: List of just construct names (for display)
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    full_texts = []
    construct_names = []
    
    # For average mode, aggregate both sources
    sections_to_load = ["oxford", "cognitive_psychology"] if section == "average" else [section]
    
    for section_name in sections_to_load:
        constructs = config.get(section_name, {})
        
        for construct_name, definitions_list in constructs.items():
            if construct_name not in construct_names:
                # Add construct name once (for averaging across sources)
                construct_names.append(construct_name)
            
            if isinstance(definitions_list, list):
                # Take the first definition if there are multiple
                definition = definitions_list[0].get('description', '') if definitions_list else ''
                
                if use_names_only:
                    # Just use the construct name
                    full_text = construct_name
                else:
                    # Use construct name + description
                    full_text = f"{construct_name}: {definition}"
                
                full_texts.append(full_text)
    
    return full_texts, construct_names


def generate_markdown_report(
    cluster_labels: pd.DataFrame,
    labels: np.ndarray,
    outpath: str
):
    """
    Generate a markdown report with cluster labels and interpretations.
    
    Args:
        cluster_labels: DataFrame with cluster information
        labels: Original cluster labels for all items
        outpath: Path to save markdown report
    """
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write("# Cluster Centroid Labeling Report\n\n")
        f.write("## Overview\n\n")
        f.write(f"- **Total clusters**: {len(cluster_labels)}\n")
        f.write(f"- **Total items**: {len(labels)}\n")
        f.write(f"- **Noise points**: {np.sum(labels == -1)}\n\n")
        
        f.write("## Cluster Interpretations\n\n")
        f.write("Each cluster's centroid is decoded by finding the most similar items ")
        f.write("from the original item set. These representative items help interpret ")
        f.write("the semantic meaning of each cluster/factor.\n\n")
        
        for _, row in cluster_labels.iterrows():
            cluster_id = row['cluster_id']
            n_items = row['n_items']
            rep_items = row['representative_items']
            top_sim = row['top_similarity']
            mean_sim = row['mean_similarity']
            
            f.write(f"### Cluster {cluster_id}\n\n")
            f.write(f"- **Size**: {n_items} items\n")
            f.write(f"- **Representative items**: {rep_items}\n")
            f.write(f"- **Top similarity to centroid**: {top_sim:.3f}\n")
            f.write(f"- **Mean similarity to centroid**: {mean_sim:.3f}\n\n")
            
            # Add nearest items details
            nearest_df = row['nearest_items_df']
            f.write("**Nearest items to centroid**:\n\n")
            for idx, item_row in nearest_df.iterrows():
                f.write(f"{idx + 1}. **{item_row['item_id']}** ")
                f.write(f"(similarity: {item_row['similarity']:.3f})\n")
                # Truncate text for readability
                text = item_row['item_text']
                if len(text) > 150:
                    text = text[:150] + "..."
                f.write(f"   - {text}\n\n")
            
            f.write("---\n\n")
    
    print(f"Markdown report saved to: {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate interpretable labels for clusters by decoding centroids"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["all-mpnet-base-v2", "all-roberta-large-v1"],
        help="Model name - sets all paths automatically and enables --decode-concepts and --visualize"
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Path to embeddings .npy file (auto-set if --model is provided)"
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        default=None,
        help="Path to CSV file with cluster labels (auto-set if --model is provided)"
    )
    parser.add_argument(
        "--items",
        type=str,
        default="data/processed/items.csv",
        help="Path to items CSV file"
    )
    parser.add_argument(
        "--model-for-concepts",
        type=str,
        default=None,
        help="Model name for concept decoding (defaults to --model if provided, else all-mpnet-base-v2)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for cluster labels (default: auto-detect from model)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest items to show per cluster"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="item_text",
        help="Column name with item text"
    )
    parser.add_argument(
        "--definition",
        type=str,
        default="oxford",
        help="Definition source(s) to use for labeling. Options: 'oxford', 'cognitive_psychology', or 'average' (averages embeddings across oxford and cognitive_psychology)"
    )
    parser.add_argument(
        "--no-def",
        action="store_true",
        help="Use only construct names (no descriptions) for concept decoding"
    )
    parser.add_argument(
        "--decode-concepts",
        action="store_true",
        help="Also decode centroids against cognitive domain concepts"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots (requires matplotlib and seaborn)"
    )
    
    args = parser.parse_args()
    
    # If --model is provided, auto-set all paths and enable features
    if args.model is not None:
        root = Path(__file__).resolve().parents[2]
        model_slug = args.model.replace("-", "_")
        
        # Determine the analysis directory from --output-dir or auto-detect
        if args.output_dir is not None:
            analysis_dir = Path(args.output_dir)
        else:
            analysis_dir = root / "data" / "outputs" / "experiments" / f"{model_slug}_analysis"
        
        # Set embeddings path if not provided (prefer run-specific over processed/)
        if args.embeddings is None:
            run_specific_emb = analysis_dir / f"embeddings_{model_slug}.npy"
            if run_specific_emb.exists():
                args.embeddings = str(run_specific_emb)
            else:
                args.embeddings = str(root / "data" / "processed" / f"embeddings_{model_slug}.npy")
            print(f"Using embeddings: {args.embeddings}")
        
        # Set labels file if not provided
        if args.labels_file is None:
            args.labels_file = str(analysis_dir / "03_clustering" / "cluster_assignments.csv")
            print(f"Using labels file: {args.labels_file}")
        
        # Set items file if not provided (prefer run-specific over processed/)
        if args.items == "data/processed/items.csv":  # Check if still default
            run_specific_items = analysis_dir / "items.csv"
            if run_specific_items.exists():
                args.items = str(run_specific_items)
            else:
                args.items = str(root / "data" / "processed" / "items.csv")
            print(f"Using items: {args.items}")
        
        # Enable decode-concepts and visualize by default
        args.decode_concepts = True
        args.visualize = True
        print(f"Auto-enabled: --decode-concepts --visualize")
        
        # Set model for concepts if not provided
        if args.model_for_concepts is None:
            args.model_for_concepts = args.model
        
        # Set output directory for labeling results (always use 04_labeling subdirectory)
        if args.output_dir is None:
            args.output_dir = str(analysis_dir / "04_labeling")
        else:
            # If user provided output_dir, check if it already ends with 04_labeling
            output_path = Path(args.output_dir)
            if output_path.name != "04_labeling":
                args.output_dir = str(output_path / "04_labeling")
    else:
        # Require embeddings and labels-file if model not provided
        if args.embeddings is None:
            parser.error("--embeddings is required when --model is not provided")
        if args.labels_file is None:
            parser.error("--labels-file is required when --model is not provided")
        
        # Set default model for concepts
        if args.model_for_concepts is None:
            args.model_for_concepts = "all-mpnet-base-v2"
    
    # Auto-detect output directory from model name if not provided
    # (only if --model was NOT provided above)
    if args.model is None and args.output_dir is None:
        # Extract model name from embeddings path
        emb_filename = Path(args.embeddings).stem
        if "all_mpnet_base_v2" in emb_filename or "all-mpnet-base-v2" in emb_filename:
            model_slug = "all_mpnet_base_v2"
        elif "all_roberta_large_v1" in emb_filename or "all-roberta-large-v1" in emb_filename:
            model_slug = "all_roberta_large_v1"
        else:
            # Fallback to generic path
            model_slug = "unknown_model"
        
        # Create output directory following pipeline structure
        root = Path(__file__).resolve().parents[2]
        args.output_dir = str(root / "data" / "outputs" / "experiments" / f"{model_slug}_analysis" / "04_labeling")
        print(f"Auto-detected output directory: {args.output_dir}")
    
    # Load data
    print("Loading data...")
    embeddings = np.load(args.embeddings)
    labels_df = pd.read_csv(args.labels_file)
    
    # Handle both 'label' and 'cluster' column names for compatibility
    if 'label' in labels_df.columns:
        labels = labels_df['label'].values
    elif 'cluster' in labels_df.columns:
        labels = labels_df['cluster'].values
    else:
        raise ValueError(f"Labels file must have 'label' or 'cluster' column. Found: {labels_df.columns.tolist()}")
    
    # Load items - handle both .txt and .csv files
    if args.items.endswith('.csv'):
        items = pd.read_csv(args.items)
    else:
        items = parse_items(args.items)
    
    print(f"Loaded {len(embeddings)} embeddings with {len(np.unique(labels))} unique clusters")

    # Sanity checks to prevent shape mismatches
    if len(embeddings) != len(labels):
        raise ValueError(
            "Embeddings count does not match labels count. "
            f"Embeddings: {len(embeddings)}, labels: {len(labels)}. "
            "Ensure you point --embeddings to the file generated for the same run as the labels, "
            "or regenerate embeddings for this item set."
        )

    if len(labels) != len(items):
        raise ValueError(
            "Labels count does not match items count. "
            f"Labels: {len(labels)}, items: {len(items)}. "
            "Verify that --items and --labels-file refer to the same pipeline run."
        )
    
    # Compute centroids
    print("\nComputing cluster centroids...")
    centroids = compute_cluster_centroids(embeddings, labels)
    print(f"Computed centroids for {len(centroids)} clusters")
    
    # Generate labels
    print("\nFinding representative items for each cluster...")
    cluster_labels = generate_cluster_labels(
        centroids,
        embeddings,
        items,
        top_k=args.top_k,
        text_column=args.text_column
    )
    
    # Save results
    print("\nSaving results...")
    save_cluster_labels(cluster_labels, args.output_dir, include_details=True)
    
    # Generate markdown report
    report_path = os.path.join(args.output_dir, 'cluster_labels_report.md')
    generate_markdown_report(cluster_labels, labels, report_path)
    
    # Optional: Decode against cognitive concepts
    if args.decode_concepts:
        print("\nDecoding centroids against cognitive domain concepts...")
        model = load_model(args.model_for_concepts)
        
        # Load candidate concepts from YAML
        root = Path(__file__).resolve().parents[2]
        yaml_path = root / "config" / "construct_definition.yaml"
        candidate_texts, construct_names = load_candidate_texts_from_yaml(
            str(yaml_path), 
            section=args.definition,
            use_names_only=args.no_def
        )
        
        # Handle "average" mode: compute embeddings separately for oxford and cognitive_psychology, then average
        if args.definition == "average":
            print("  Using 'average' mode: averaging embeddings across definition sources...")
            # Load both sources separately
            oxford_texts, oxford_names = load_candidate_texts_from_yaml(
                str(yaml_path), 
                section="oxford",
                use_names_only=args.no_def
            )
            cog_texts, cog_names = load_candidate_texts_from_yaml(
                str(yaml_path), 
                section="cognitive_psychology",
                use_names_only=args.no_def
            )
            
            # Embed both
            oxford_df = pd.DataFrame({'text': oxford_texts})
            oxford_embeddings = embed_items(oxford_df, text_column='text', model=model)
            
            cog_df = pd.DataFrame({'text': cog_texts})
            cog_embeddings = embed_items(cog_df, text_column='text', model=model)
            
            # Average them - assuming they have same constructs in same order
            candidate_embeddings = (oxford_embeddings + cog_embeddings) / 2.0
        else:
            # Standard mode: embed as loaded
            candidate_df = pd.DataFrame({'text': candidate_texts})
            candidate_embeddings = embed_items(
                candidate_df, 
                text_column='text', 
                model=model
            )
        
        # Build similarity matrix: clusters x concepts
        sorted_cluster_ids = sorted(centroids.keys())
        similarity_matrix = np.zeros((len(sorted_cluster_ids), len(construct_names)))
        
        for i, cluster_id in enumerate(sorted_cluster_ids):
            centroid = centroids[cluster_id]
            similarities = cosine_similarity(
                centroid.reshape(1, -1), 
                candidate_embeddings
            )[0]
            similarity_matrix[i] = similarities
        
        # Greedy assignment: assign each concept to the cluster that wants it most
        # (ensuring each concept is assigned to at most one cluster)
        assigned_concepts = set()
        cluster_concept_assignment = {}
        
        # Iterate from strongest to weakest similarity
        flat_indices = np.argsort(-similarity_matrix.ravel())
        for flat_idx in flat_indices:
            cluster_idx, concept_idx = np.unravel_index(flat_idx, similarity_matrix.shape)
            cluster_id = sorted_cluster_ids[cluster_idx]
            
            # Assign if concept not yet used and cluster not yet assigned
            if concept_idx not in assigned_concepts and cluster_id not in cluster_concept_assignment:
                cluster_concept_assignment[cluster_id] = (concept_idx, similarity_matrix[cluster_idx, concept_idx])
                assigned_concepts.add(concept_idx)
        
        # Build results - using construct_names for display only
        concept_results = []
        for cluster_id in sorted_cluster_ids:
            if cluster_id in cluster_concept_assignment:
                concept_idx, similarity = cluster_concept_assignment[cluster_id]
                top_concept = construct_names[concept_idx]  # Use name, not full text
                # Get all similarities for this cluster for reporting
                all_sims = similarity_matrix[sorted_cluster_ids.index(cluster_id)]
                all_concepts_str = '; '.join([
                    f"{construct_names[i]} ({all_sims[i]:.3f})" 
                    for i in range(len(construct_names))
                ])
            else:
                # Fallback if not assigned (shouldn't happen with enough concepts)
                top_concept = "unassigned"
                similarity = 0.0
                all_concepts_str = ""
            
            concept_results.append({
                'cluster_id': cluster_id,
                'top_concept': top_concept,
                'top_concept_similarity': similarity,
                'all_concepts': all_concepts_str
            })
        
        concept_df = pd.DataFrame(concept_results)
        concept_df.to_csv(
            os.path.join(args.output_dir, 'cluster_concept_labels.csv'),
            index=False
        )
        print(f"Concept labels saved to: {os.path.join(args.output_dir, 'cluster_concept_labels.csv')}")
    else:
        concept_df = None
    
    # Optional: Generate visualizations
    if args.visualize:
        try:
            from bayley_nlp.core.labeling_viz import create_all_visualizations
            create_all_visualizations(
                centroids=centroids,
                embeddings=embeddings,
                labels=labels,
                items=items,
                outdir=args.output_dir,
                concept_results=concept_df,
                top_k=args.top_k
            )
        except ImportError as e:
            print(f"\nWarning: Could not generate visualizations: {e}")
            print("Install matplotlib and seaborn to enable visualizations.")
    
    print("\n✓ Cluster labeling complete!")
    print(f"\nOutputs saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
