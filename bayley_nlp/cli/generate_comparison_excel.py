
# --- UTF-8 stdout/stderr: this script prints Unicode (arrows, checkmarks)
# which crashes under Windows' default cp1252 encoding; force UTF-8. ---
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")
del _stream

#!/usr/bin/env python
"""
Generate Excel summary from item comparison results and extended clustering.

Supports two modes:
1. Standard mode: Takes item_comparison_theoretical.csv from Step 05
   - All columns from item_comparison_theoretical.csv
   - Item name and description columns
   - Reference items sorted first, then remaining items

2. Extended mode: Includes new items from extended clustering (03_clustering_extended_kprecomputed)
   - Combines reference items with their comparison results
   - Adds new items with their cluster assignments
   - All reference items first, then new items

Usage:
    python bayley_nlp/cli/generate_comparison_excel.py --folder-path "path/to/output/folder"
    python bayley_nlp/cli/generate_comparison_excel.py --folder-path data/outputs/step2_b4_validation/all_mpnet_base_v2_kmeans_consensus_k_precomputed_no_dr
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from openpyxl.styles import PatternFill


def load_centroids_from_clustering(clustering_data_path: Path, model_slug: str) -> np.ndarray:
    """
    Reconstruct centroids from clustering results.
    
    Args:
        clustering_data_path: Path to 03_clustering directory
        model_slug: Model name slug for embedding file
        
    Returns:
        Centroids array (n_clusters, embedding_dim)
    """
    try:
        # Load cluster assignments
        cluster_assigns = pd.read_csv(clustering_data_path / "cluster_assignments.csv")
        embeddings = np.load(clustering_data_path.parent / f"embeddings_{model_slug}.npy")
        
        n_clusters = len(cluster_assigns['cluster'].unique())
        n_features = embeddings.shape[1]
        centroids = np.zeros((n_clusters, n_features))
        
        for cluster_id in range(n_clusters):
            mask = cluster_assigns['cluster'] == cluster_id
            if mask.sum() > 0:
                centroids[cluster_id] = embeddings[mask].mean(axis=0)
        
        return centroids
    except Exception as e:
        print(f"      Warning: Could not load centroids: {e}")
        return None


def compute_inverse_distances(result_df: pd.DataFrame, folder_path: Path, model_slug: str) -> pd.Series:
    """
    Compute inverse distance to centroid for all items.
    
    Inverse distance = 1 / (1 + cosine_distance)
    Higher values = closer to centroid (better confidence)
    
    Args:
        result_df: Dataframe with item data including cluster assignments
        folder_path: Base output folder
        model_slug: Model name slug for embedding file
        
    Returns:
        Series of inverse distances indexed by item_id
    """
    print("      Computing inverse distance to centroid...")
    
    extended_clustering_path = folder_path / "03_clustering_extended_kprecomputed"
    clustering_path = folder_path / "03_clustering"
    
    try:
        # Load embeddings
        if extended_clustering_path.exists():
            emb_file = extended_clustering_path / f"embeddings_{model_slug}_combined.npy"
            embeddings = np.load(emb_file)
        else:
            emb_file = folder_path / f"embeddings_{model_slug}.npy"
            embeddings = np.load(emb_file)
        
        # Load centroids
        centroids = load_centroids_from_clustering(clustering_path, model_slug)
        if centroids is None:
            print("      Warning: Could not compute inverse distances (missing centroids)")
            return pd.Series(dtype=float)
        
        # Compute cosine distances
        distances = cosine_distances(embeddings, centroids)
        
        # Get assigned cluster for each item
        cluster_col = None
        if 'cluster' in result_df.columns:
            cluster_col = result_df['cluster'].values
        
        if cluster_col is None:
            print("      Warning: No cluster assignments found")
            return pd.Series(dtype=float)
        
        # Get distance to assigned centroid for each item
        item_distances = np.array([
            distances[i, int(cluster_col[i])] 
            for i in range(len(cluster_col))
        ])
        
        # Compute inverse distance (higher = better)
        inverse_distances = 1.0 / (1.0 + item_distances)
        
        print(f"      Computed inverse distances (mean: {inverse_distances.mean():.4f}, std: {inverse_distances.std():.4f})")
        return pd.Series(inverse_distances, index=result_df.index)
        
    except Exception as e:
        print(f"      Warning: Error computing inverse distances: {e}")
        return pd.Series(dtype=float)


def load_cluster_labels(labels_path: Path) -> Dict[int, str]:
    """
    Load cluster labels from cluster_concept_labels.csv.
    
    Returns:
        Dict mapping cluster_id to label (top_concept).
    """
    if not labels_path.exists():
        print(f"      Warning: cluster labels file not found: {labels_path}")
        return {}

    labels_df = pd.read_csv(labels_path)
    if 'cluster_id' in labels_df.columns and 'top_concept' in labels_df.columns:
        return dict(zip(labels_df['cluster_id'], labels_df['top_concept']))

    if 'cluster' in labels_df.columns and 'label' in labels_df.columns:
        return dict(zip(labels_df['cluster'], labels_df['label']))

    print(f"      Warning: Unsupported label format in {labels_path}, detected columns: {list(labels_df.columns)}")
    return {}


def find_items_metadata() -> pd.DataFrame:
    """
    Load item metadata from the processed items.csv file.
    
    This file contains item_id, item_title, item_materials, item_description, etc.
    """
    # Try to find items.csv in common locations
    possible_paths = [
        Path(__file__).resolve().parents[2] / "data" / "processed" / "items.csv",
        Path.cwd() / "data" / "processed" / "items.csv",
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"      Loading item metadata from: {path}")
            return pd.read_csv(path)
    
    raise FileNotFoundError(
        "Could not find items.csv. Searched:\n" +
        "\n".join(f"        {p}" for p in possible_paths)
    )


def identify_reference_items(comparison_df: pd.DataFrame) -> set:
    """
    Identify which items are reference items.
    
    Checks for either:
    - reference_theoretical column is not null (from theoretical comparison)
    - reference_empirical column is not null (from empirical comparison)
    - is_reference column == True (from extended clustering)
    """
    reference_set = set()
    
    # Check for theoretical reference
    if 'reference_theoretical' in comparison_df.columns:
        reference_set.update(comparison_df[comparison_df['reference_theoretical'].notna()]['item_id'].unique())
    
    # Check for empirical reference
    if 'reference_empirical' in comparison_df.columns:
        reference_set.update(comparison_df[comparison_df['reference_empirical'].notna()]['item_id'].unique())
    
    # Check for is_reference column
    if 'is_reference' in comparison_df.columns:
        reference_set.update(comparison_df[comparison_df['is_reference'] == True]['item_id'].unique())
    
    return reference_set



def extract_model_slug(folder_path: Path) -> str:
    """
    Extract model slug from folder path.
    
    Example: 
        all_mpnet_base_v2_kmeans_consensus_k_precomputed -> all_mpnet_base_v2
        all_mpnet_base_v2_kmeans_consensus_k_precomputed_no_dr -> all_mpnet_base_v2
    """
    folder_name = folder_path.name
    # Remove common suffixes to get the model slug (order matters - longest first)
    suffixes = [
        '_kmeans_consensus_k_precomputed_no_dr',
        '_kmeans_consensus_k_precomputed',
        '_kmeans_consensus_k_no_dr',
        '_kmeans_consensus_k',
        '_precomputed_no_dr',
        '_precomputed',
        '_no_dr',
        '_kmeans'
    ]
    for suffix in suffixes:
        if folder_name.endswith(suffix):
            return folder_name.replace(suffix, '')
    return folder_name


def generate_excel_summary(folder_path: Path, output_path: Path = None):
    """
    Generate Excel summary from item comparison results and/or extended clustering.
    
    Args:
        folder_path: Path to the output folder (e.g., all_mpnet_base_v2_kmeans_consensus_k_precomputed)
        output_path: Optional path for output Excel file. Defaults to folder_path/comparison_summary.xlsx
    """
    folder_path = Path(folder_path)
    
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Extract model slug from folder path
    model_slug = extract_model_slug(folder_path)
    print(f"Using model slug: {model_slug}\n")
    
    print(f"\n{'='*70}")

    print("GENERATE COMPARISON EXCEL SUMMARY")
    print(f"{'='*70}\n")
    print(f"Input folder: {folder_path}\n")
    
    # Check for extended clustering results first
    extended_clustering_path = folder_path / "03_clustering_extended_kprecomputed"
    use_extended = extended_clustering_path.exists()
    
    if use_extended:
        print("[MODE] Extended clustering (includes new items)\n")
        cluster_file = extended_clustering_path / "cluster_assignments_combined.csv"
        items_metadata_file = extended_clustering_path / "items_combined.csv"
        theoretical_file = folder_path / "05_model_comparison" / "item_comparison_theoretical.csv"
        empirical_file = folder_path / "05_model_comparison" / "item_comparison_empirical.csv"
        labels_file = folder_path / "04_labeling" / "cluster_concept_labels.csv"
        
        if not cluster_file.exists():
            raise FileNotFoundError(f"Could not find {cluster_file}")
        
        print("[1/6] Loading extended clustering results...")
        cluster_df = pd.read_csv(cluster_file)
        print(f"      Loaded {len(cluster_df)} items (reference + new)")

        print("\n[2/6] Loading comparison results (theoretical and empirical)...")
        theoretical_df = pd.DataFrame()
        empirical_df = pd.DataFrame()
        
        if theoretical_file.exists():
            theoretical_df = pd.read_csv(theoretical_file)
            # Keep item_id, reference_factor, and agrees for agreement tracking
            cols_to_keep = ['item_id', 'reference_factor']
            if 'agrees' in theoretical_df.columns:
                cols_to_keep.append('agrees')
            theoretical_df = theoretical_df[cols_to_keep].copy()
            theoretical_df.rename(columns={'reference_factor': 'reference_theoretical'}, inplace=True)
            print(f"      Loaded {len(theoretical_df)} theoretical comparison rows")
        else:
            print("      No item_comparison_theoretical.csv found")
            
        if empirical_file.exists():
            empirical_df = pd.read_csv(empirical_file)
            # Extract empirical reference factor from reference_factor column
            empirical_df = empirical_df[['item_id', 'reference_factor']].copy()
            empirical_df.rename(columns={'reference_factor': 'reference_empirical'}, inplace=True)
            print(f"      Loaded {len(empirical_df)} empirical comparison rows")
        else:
            print("      No item_comparison_empirical.csv found")

        print("\n[3/6] Loading cluster label mappings...")
        cluster_label_dict = load_cluster_labels(labels_file)
        if cluster_label_dict:
            print(f"      Loaded {len(cluster_label_dict)} cluster labels")
        else:
            print("      No cluster labels available; detected_factor will be blank")
        
        print("\n[4/6] Loading item metadata...")

        if items_metadata_file.exists():
            items_metadata = pd.read_csv(items_metadata_file)
            available_cols = ['item_id']
            if 'item_title' in items_metadata.columns:
                available_cols.append('item_title')
            if 'item_description' in items_metadata.columns:
                available_cols.append('item_description')
            if 'scoring_json' in items_metadata.columns:
                available_cols.append('scoring_json')
            items_metadata = items_metadata[available_cols]
            
            if 'item_title' in items_metadata.columns:
                items_metadata.rename(columns={'item_title': 'name'}, inplace=True)
            if 'item_description' in items_metadata.columns:
                items_metadata.rename(columns={'item_description': 'description'}, inplace=True)
            if 'scoring_json' in items_metadata.columns:
                items_metadata.rename(columns={'scoring_json': 'scoring_criteria'}, inplace=True)
                
            print(f"      Loaded metadata for {len(items_metadata)} items from extended data")
        else:
            items_metadata = None
            print(f"      No items_combined.csv found, will use fallback metadata")
        
        print("\n[5/6] Merging extended results, reference comparison, and metadata...")
        result_df = cluster_df.copy()

        # Drop item_text if present
        if 'item_text' in result_df.columns:
            result_df = result_df.drop(columns=['item_text'])

        # Merge theoretical comparison fields into full item list
        if len(theoretical_df) > 0:
            result_df = result_df.merge(theoretical_df, on='item_id', how='left')
        
        # Merge empirical comparison fields into full item list
        if len(empirical_df) > 0:
            result_df = result_df.merge(empirical_df, on='item_id', how='left')

        # Add detected_factor from cluster labels where missing
        if cluster_label_dict:
            if 'detected_factor' not in result_df.columns:
                result_df['detected_factor'] = result_df['cluster'].map(cluster_label_dict)
            else:
                result_df['detected_factor'] = result_df['detected_factor'].fillna(
                    result_df['cluster'].map(cluster_label_dict)
                )
        
        # Ensure reference_factor exists and is blank for non-reference
        if 'reference_factor' not in result_df.columns:
            result_df['reference_factor'] = None
        if 'is_reference' in result_df.columns:
            result_df.loc[result_df['is_reference'] == False, 'reference_factor'] = None
            result_df.loc[result_df['is_reference'] == False, 'agrees'] = None

        # Merge metadata
        if items_metadata is not None:
            result_df = result_df.merge(items_metadata, on='item_id', how='left')
        else:
            cols_to_load = ['item_id', 'item_title', 'item_description']
            default_items_full = find_items_metadata()
            if 'scoring_json' in default_items_full.columns:
                cols_to_load.append('scoring_json')
            default_items = default_items_full[cols_to_load].copy()
            rename_dict = {
                'item_title': 'name',
                'item_description': 'description'
            }
            if 'scoring_json' in cols_to_load:
                rename_dict['scoring_json'] = 'scoring_criteria'
            default_items.rename(columns=rename_dict, inplace=True)
            result_df = result_df.merge(default_items, on='item_id', how='left')
        
        # Compute inverse distance to centroid
        print("\n[6/6] Computing confidence scores (inverse distance to centroid)...")
        inverse_distances = compute_inverse_distances(result_df, folder_path, model_slug)
        if len(inverse_distances) > 0:
            result_df['inverse_distance_to_centroid'] = inverse_distances.values
        
    else:
        print("[MODE] Standard comparison (reference items only)\n")
        comparison_file = folder_path / "05_model_comparison" / "item_comparison_theoretical.csv"
        labels_file = folder_path / "04_labeling" / "cluster_concept_labels.csv"
        
        if not comparison_file.exists():
            raise FileNotFoundError(
                f"Could not find item_comparison_theoretical.csv at {comparison_file}\n"
                f"Expected folder structure: {folder_path}/05_model_comparison/"
            )
        
        print(f"Comparison file: {comparison_file}\n")
        
        print("[1/4] Loading comparison results...")
        comparison_df = pd.read_csv(comparison_file)
        print(f"      Loaded {len(comparison_df)} items")
        
        print("\n[2/4] Loading item metadata...")
        items_metadata_full = find_items_metadata()
        cols_to_load = ['item_id', 'item_title', 'item_description']
        if 'scoring_json' in items_metadata_full.columns:
            cols_to_load.append('scoring_json')
        items_metadata = items_metadata_full[cols_to_load].copy()
        rename_dict = {
            'item_title': 'name',
            'item_description': 'description'
        }
        if 'scoring_json' in cols_to_load:
            rename_dict['scoring_json'] = 'scoring_criteria'
        items_metadata.rename(columns=rename_dict, inplace=True)
        print(f"      Loaded metadata for {len(items_metadata)} items")
        
        print("\n[3/4] Loading cluster label mappings...")
        cluster_label_dict = load_cluster_labels(labels_file)
        if cluster_label_dict and 'detected_factor' not in comparison_df.columns:
            comparison_df['detected_factor'] = comparison_df['cluster'].map(cluster_label_dict)
        
        print("\n[4/4] Merging comparison results with item metadata...")
        result_df = comparison_df.merge(items_metadata, on='item_id', how='left')
        
        # Compute inverse distance to centroid
        print("\n[5/5] Computing confidence scores (inverse distance to centroid)...")
        inverse_distances = compute_inverse_distances(result_df, folder_path, model_slug)
        if len(inverse_distances) > 0:
            result_df['inverse_distance_to_centroid'] = inverse_distances.values

        
    print(f"      Merged {len(result_df)} items")

    # Identify and sort reference items first
    print("\nSorting items (reference items first)...")
    reference_items = identify_reference_items(result_df)
    print(f"      Found {len(reference_items)} reference items")
    print(f"      Found {len(result_df) - len(reference_items)} new items")
    
    # Create a sort key: reference items first (0), then others (1)
    result_df['_is_reference'] = result_df['item_id'].apply(lambda x: 0 if x in reference_items else 1)
    
    # Sort by reference status, then reference_factor, then item_id
    sort_cols = ['_is_reference']
    if 'reference_theoretical' in result_df.columns:
        sort_cols.append('reference_theoretical')
    sort_cols.append('item_id')
    result_df = result_df.sort_values(sort_cols, na_position='last').reset_index(drop=True)
    
    # Drop the temporary sorting column
    result_df = result_df.drop('_is_reference', axis=1)
    
    # Drop cluster and reference_factor columns before output (keep detected_factor and reference_empirical/theoretical instead)
    columns_to_drop = []
    if 'cluster' in result_df.columns:
        columns_to_drop.append('cluster')
    if 'reference_factor' in result_df.columns:
        columns_to_drop.append('reference_factor')
    if columns_to_drop:
        result_df = result_df.drop(columns=columns_to_drop)

    # Reorder columns: item_id, name, description, scoring_criteria first, then key comparison fields
    main_cols = [
        'item_id',
        'name',
        'description',
        'scoring_criteria',
        'reference_empirical',
        'reference_theoretical',
        'detected_factor',
        'agrees',
        'inverse_distance_to_centroid'
    ]
    # Filter to only columns that exist
    main_cols = [col for col in main_cols if col in result_df.columns]
    other_cols = [col for col in result_df.columns if col not in main_cols]
    result_df = result_df[main_cols + other_cols]
    
    # Set output path
    if output_path is None:
        output_path = folder_path / "comparison_summary.xlsx"
    else:
        output_path = Path(output_path)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to Excel
    print(f"\nWriting Excel file: {output_path}")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        result_df.to_excel(writer, sheet_name='Item Comparison', index=False)
        
        # Auto-adjust column widths
        workbook = writer.book
        worksheet = writer.sheets['Item Comparison']
        
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            
            for cell in column:
                try:
                    if cell.value:
                        cell_length = len(str(cell.value))
                        if cell_length > max_length:
                            max_length = cell_length
                except:
                    pass
            
            # Set column width with some padding
            adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"      ✓ Excel file created successfully")
    print(f"\nSummary Statistics:")
    print(f"  - Total items: {len(result_df)}")
    print(f"  - Reference items: {len(reference_items)}")
    print(f"  - New items: {len(result_df) - len(reference_items)}")
    
    # Show agreement statistics if applicable
    if 'agrees' in result_df.columns:
        agrees_count = result_df['agrees'].sum()
        agrees_pct = (agrees_count / len(result_df[result_df['is_reference']==True])) * 100
        print(f"  - Items agreeing with reference: {agrees_count} ({agrees_pct:.1f}%)")
    
    print(f"\n{'='*70}")
    print(f"✓ EXCEL SUMMARY COMPLETE!")
    print(f"{'='*70}\n")
    
    return result_df


def generate_comparison_merged(parent_folder: Path, output_path: Path = None):
    """
    Generate merged comparison Excel from two subfolder summaries.
    
    This function:
    1. Finds two subfolders in parent_folder
    2. Generates comparison_summary.xlsx for each subfolder
    3. Merges the results into a single file with conditional formatting
    
    Args:
        parent_folder: Path to parent folder containing two model output subfolders
        output_path: Optional path for merged output. Defaults to parent_folder/comparison_merged.xlsx
    """
    parent_folder = Path(parent_folder)
    
    if not parent_folder.is_dir():
        raise FileNotFoundError(f"Parent folder not found: {parent_folder}")
    
    print(f"\n{'='*70}")
    print("GENERATE MERGED COMPARISON EXCEL")
    print(f"{'='*70}\n")
    print(f"Parent folder: {parent_folder}\n")
    
    # Find subfolders (look for folders with 'kmeans' in the name)
    subfolders = [f for f in parent_folder.iterdir() if f.is_dir() and 'kmeans' in f.name.lower()]
    
    if len(subfolders) < 2:
        raise ValueError(
            f"Expected at least 2 subfolders in {parent_folder}, found {len(subfolders)}.\n"
            f"Subfolders: {[f.name for f in subfolders]}"
        )
    
    # Sort to ensure consistent order (typically: with DR first, no DR second)
    subfolders = sorted(subfolders, key=lambda x: ('no_dr' in x.name.lower(), x.name))
    
    print(f"Found {len(subfolders)} subfolders:")
    for i, folder in enumerate(subfolders[:2]):
        print(f"  [{i+1}] {folder.name}")
    
    # Use first two folders
    folder1, folder2 = subfolders[0], subfolders[1]
    
    # Generate individual summaries
    print(f"\n[1/3] Generating summary for: {folder1.name}")
    df1 = generate_excel_summary(folder1)
    
    print(f"\n[2/3] Generating summary for: {folder2.name}")
    df2 = generate_excel_summary(folder2)
    
    print(f"\n[3/3] Merging results...")
    
    # Determine labels for detected_factor and agrees columns
    label1 = "Clustering After Dimension Reduction" if 'no_dr' not in folder1.name.lower() else "Clustering in Embedding Space"
    label2 = "Clustering After Dimension Reduction" if 'no_dr' not in folder2.name.lower() else "Clustering in Embedding Space"
    
    # Prepare dataframes for merging
    # Keep only necessary columns from each
    base_cols = ['item_id', 'name', 'description', 'scoring_criteria', 'reference_empirical', 'reference_theoretical']
    
    # Get base columns from first dataframe
    merge_cols = [col for col in base_cols if col in df1.columns]
    merged_df = df1[merge_cols].copy()
    
    # Add detected_factor and agrees from first dataframe
    if 'detected_factor' in df1.columns:
        merged_df[f'detected_factor ({label1})'] = df1['detected_factor']
    if 'agrees' in df1.columns:
        merged_df[f'agrees ({label1})'] = df1['agrees']
    
    # Prepare second dataframe columns to merge
    df2_merge = df2[['item_id']].copy()
    if 'detected_factor' in df2.columns:
        df2_merge[f'detected_factor ({label2})'] = df2['detected_factor']
    if 'agrees' in df2.columns:
        df2_merge[f'agrees ({label2})'] = df2['agrees']
    
    # Merge on item_id
    merged_df = merged_df.merge(df2_merge, on='item_id', how='outer')
    
    # Sort by reference_theoretical value (grouping same factors together), then by item_id
    # Items without reference_theoretical will be placed at the end
    if 'reference_theoretical' in merged_df.columns:
        # Fill NaN with empty string for sorting (will appear last)
        merged_df['_ref_sort'] = merged_df['reference_theoretical'].fillna('zzz_no_reference')
        merged_df = merged_df.sort_values(['_ref_sort', 'item_id']).reset_index(drop=True)
        merged_df = merged_df.drop('_ref_sort', axis=1)
    else:
        merged_df = merged_df.sort_values('item_id').reset_index(drop=True)
    
    print(f"      Merged {len(merged_df)} items")
    
    # Set output path
    if output_path is None:
        output_path = parent_folder / "comparison_merged.xlsx"
    else:
        output_path = Path(output_path)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to Excel with conditional formatting
    print(f"\nWriting merged Excel file: {output_path}")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        merged_df.to_excel(writer, sheet_name='Merged Comparison', index=False)
        
        # Get worksheet
        workbook = writer.book
        worksheet = writer.sheets['Merged Comparison']
        
        # Define fill colors
        green_fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")  # Light green
        red_fill = PatternFill(start_color="FFB6C1", end_color="FFB6C1", fill_type="solid")    # Light red
        yellow_fill = PatternFill(start_color="FFFFE0", end_color="FFFFE0", fill_type="solid")  # Light yellow
        
        # Find agrees columns indices (1-indexed for openpyxl)
        agrees_col1 = None
        agrees_col2 = None
        for idx, col in enumerate(merged_df.columns, start=1):
            if col == f'agrees ({label1})':
                agrees_col1 = idx
            elif col == f'agrees ({label2})':
                agrees_col2 = idx
        
        # Apply conditional formatting to rows (skip header row)
        if agrees_col1 is not None and agrees_col2 is not None:
            for row_idx in range(2, len(merged_df) + 2):  # Start from 2 (after header)
                agrees1_val = merged_df.iloc[row_idx - 2][f'agrees ({label1})']
                agrees2_val = merged_df.iloc[row_idx - 2][f'agrees ({label2})']
                
                # Determine fill color
                fill = None
                if pd.notna(agrees1_val) and pd.notna(agrees2_val):
                    # Both have values
                    if agrees1_val != agrees2_val:
                        fill = yellow_fill  # Different values
                    elif agrees1_val == True:
                        fill = green_fill   # Both TRUE
                    else:
                        fill = red_fill     # Both FALSE
                elif pd.notna(agrees1_val):
                    # Only first has value
                    if agrees1_val == True:
                        fill = green_fill
                    else:
                        fill = red_fill
                
                # Apply fill to entire row
                if fill is not None:
                    for col_idx in range(1, len(merged_df.columns) + 1):
                        worksheet.cell(row=row_idx, column=col_idx).fill = fill
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            
            for cell in column:
                try:
                    if cell.value:
                        cell_length = len(str(cell.value))
                        if cell_length > max_length:
                            max_length = cell_length
                except:
                    pass
            
            # Set column width with some padding
            adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"      ✓ Merged Excel file created successfully")
    print(f"\nConditional Formatting Applied:")
    print(f"  - Green: agrees ({label1}) = TRUE")
    print(f"  - Red: agrees ({label1}) = FALSE")
    print(f"  - Yellow: Different values between agrees columns")
    
    print(f"\n{'='*70}")
    print(f"✓ MERGED COMPARISON COMPLETE!")
    print(f"{'='*70}\n")
    
    return merged_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate Excel summary from item comparison results"
    )
    parser.add_argument(
        "--folder-path",
        type=str,
        required=True,
        help="Path to the output folder from pipeline run "
             "(e.g., data/outputs/step2_b4_validation/all_mpnet_base_v2_kmeans_consensus_k_precomputed) "
             "or parent folder containing multiple model outputs when using --comparison"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional output path for Excel file. Defaults to folder_path/comparison_summary.xlsx"
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Generate merged comparison from multiple subfolders. "
             "When set, folder-path should point to parent directory containing model output subfolders."
    )
    
    args = parser.parse_args()
    
    try:
        if args.comparison:
            generate_comparison_merged(args.folder_path, args.output_path)
        else:
            generate_excel_summary(args.folder_path, args.output_path)
    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
