
# --- UTF-8 stdout/stderr: this script prints Unicode (arrows, checkmarks)
# which crashes under Windows' default cp1252 encoding; force UTF-8. ---
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")
del _stream

#!/usr/bin/env python
"""
Objective 2 -- Unsupervised validation of the NLP approach on the Bayley-4.

Consensus K-Means (1,000 iterations) is applied to the embeddings of the 39
Bayley-4 items assigned by Aylward et al. (2022), clusters are labelled post
hoc by the domain name closest to each cluster centroid, and the solution is
compared with the expert-derived and empirically derived reference models
(ARI, NMI, confusion matrices, Sankey diagrams).

Sub-steps (each is a CLI script in ``bayley_nlp/cli``):
    01  embed the 39 items + dimensionality assessment
    03  consensus K-Means clustering (1,000 runs, k = 5)
    04  post-hoc cluster labelling (nearest domain name in embedding space)
    05  comparison with the Aylward et al. (2022) reference models

The manuscript reports the run WITHOUT dimensionality reduction before
clustering (output folder ``..._k_precomputed_no_dr``).  A variant that reduces
the embeddings with UMAP before every K-Means run, plus the legacy greedy
Bayley-4 -> Bayley-III mapping steps (06/06b/07) and the extension of the
clustering to all 81 Bayley-4 items, are still available behind flags but are
not used in the paper.

Usage:
    python -m bayley_nlp.pipelines.step2_b4_validation            # paper configuration
    python -m bayley_nlp.pipelines.step2_b4_validation --from-cache
        # re-run only the comparison (05) on the committed embeddings/clusters
    python -m bayley_nlp.pipelines.step2_b4_validation --with-dr --legacy-b4-b3
        # everything the original exploratory pipeline did
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bayley_nlp.core import parse_items, load_model, embed_items, ensure_dir
from bayley_nlp.core.dimensionality_reduction import get_reducer
from bayley_nlp.core.visualization_utils import (
    create_extended_visualization,
    create_cluster_change_visualization
)



# ============================================================================
# CONFIGURATION - Edit these values to change pipeline settings
# ============================================================================

MODEL = "all-mpnet-base-v2"  # Options: "all-mpnet-base-v2", "all-roberta-large-v1"
B4_ALL_ITEMS = _REPO_ROOT / "data/raw/items_description/b4_cog_items.txt"  # Full set of B4 items (81 total) - used for extension step
ITEMS_FILE_B4 = "paper_items_b4.txt"  # Filtered to reference model items only (resolved under data/raw/items_description by cli/01)
ITEMS_FILE_B3 = _REPO_ROOT / "data/raw/items_description/b3_cog_items.txt"  # B3 has no reference model, uses full set
STRATEGY = "kmeans_consensus"  # Clustering strategy ["kmeans_consensus", "kmeans_simple", "hdbscan", "kmeans_multi_init", "similarity"]
N_CLUSTERS = 5  # Number of clusters for KMeans
ASSIGNMENT_METHOD = "k_precomputed" # Method for assigning B4 extra-items to B4 clusters ["k_precomputed", "nearest_centroid"]

# Skip flags. They are set from the command line in main(); the defaults run
# the complete pipeline from the raw item text.  ``--from-cache`` sets the
# first three to True so that only the model comparison (05) is recomputed on
# the committed embeddings / cluster assignments.
SKIP_EMBED = False       # Skip Step 01 (embeddings)
SKIP_CLUSTER = False     # Skip Step 03 (clustering)
SKIP_LABEL = False       # Skip Step 04 (labeling)
SKIP_COMPARE = False     # Skip Step 05 (model comparison)
SKIP_BAYLEY_MAP = True   # Skip Steps 06/06b/07 (legacy greedy B4->B3 mapping; not used in the paper)
RUN_EXTENSION = False    # Extend the clustering to all 81 B4 items (not used in the paper)

# Directories
outpath = _REPO_ROOT / "data/outputs/step2_b4_validation"
OUTDIR_B4_DR = str(outpath / f"{MODEL.replace('-', '_')}_{STRATEGY}_{ASSIGNMENT_METHOD}")
OUTDIR_B4_NO_DR = str(outpath / f"{MODEL.replace('-', '_')}_{STRATEGY}_{ASSIGNMENT_METHOD}_no_dr")
# ============================================================================


def run_command(cmd, step_name):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"RUNNING: {step_name}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    # Ensure spawned CLI scripts can import the top-level ``bayley_nlp`` package
    # regardless of the current working directory.
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_REPO_ROOT) + (os.pathsep + existing if existing else "")
    result = subprocess.run(cmd, capture_output=False, text=True, env=env)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {step_name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    
    print(f"\n✓ {step_name} completed successfully!")
    return result


def run_independent_b3_clustering(outdir_b4: str, dr: bool = True):
    """
    Run independent clustering on PAIRED B3 items only.
    
    This ensures fair comparison in Step 07:
    - Only clusters the B3 items that were successfully paired with B4 items
    - Uses the same clustering parameters as B4 (n_clusters, strategy, etc.)
    """
    from bayley_nlp.core.clustering import KMeansConsensusStrategy
    
    print(f"\n{'='*70}")
    print("STEP 06b: INDEPENDENT B3 CLUSTERING (PAIRED ITEMS ONLY)")
    print(f"{'='*70}\n")
    
    outdir = Path(outdir_b4)
    b3_analysis_dir = ensure_dir(str(outdir / "b3_independent_analysis"))
    b3_clustering_dir = ensure_dir(str(b3_analysis_dir / "03_clustering"))
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Load pairing results to get list of paired B3 items
    print("[1/4] Loading pairing results...")
    pairs_file = outdir / "06_b4_b3_pairing" / "b4_b3_greedy_pairs.csv"
    if not pairs_file.exists():
        raise FileNotFoundError(f"Pairing file not found: {pairs_file}\nRun Step 06 first.")
    
    pairs_df = pd.read_csv(pairs_file)
    paired_b3_ids = pairs_df['b3_item_id'].tolist()
    print(f"      Found {len(paired_b3_ids)} paired B3 items")
    
    # Load all B3 items and filter to paired ones
    print("\n[2/4] Loading and filtering B3 items...")
    items_all_b3 = parse_items(str(ITEMS_FILE_B3))
    items_b3_paired = items_all_b3[items_all_b3['item_id'].isin(paired_b3_ids)].copy().reset_index(drop=True)
    print(f"      Filtered to {len(items_b3_paired)} paired B3 items")
    
    # Save filtered items
    items_b3_paired.to_csv(b3_analysis_dir / "items.csv", index=False)
    
    # Generate embeddings for paired B3 items
    print("\n[3/4] Generating embeddings for paired B3 items...")
    model = load_model(MODEL)
    emb_b3_paired = embed_items(items_b3_paired, text_column="item_text", model=model)
    print(f"      Generated embeddings (shape: {emb_b3_paired.shape})")
    
    # Save embeddings
    model_slug = MODEL.replace("/", "_").replace("-", "_")
    np.save(b3_analysis_dir / f"embeddings_{model_slug}.npy", emb_b3_paired)
    
    # Run clustering on paired B3 items
    print("\n[4/4] Running independent clustering on paired B3 items...")
    print(f"      Strategy: {STRATEGY}")
    print(f"      Number of clusters: {N_CLUSTERS}")
    print(f"      Dimensionality reduction: {'Yes' if dr else 'No'}")
    
    umap_kwargs = {
        "metric": "cosine",
        "n_neighbors": 10,
        "min_dist": 0.0,
        "n_components": 15,
    }
    
    strategy = KMeansConsensusStrategy(
        embeddings=emb_b3_paired,
        items=items_b3_paired,
        output_dir=b3_clustering_dir,
        model_name=MODEL,
        timestamp=timestamp,
        n_clusters=N_CLUSTERS,
        n_iterations=1000,
        umap_kwargs=umap_kwargs,
        consensus_linkage='average',
        min_stability=0.2,
        use_dr=dr,
        dr_method='umap',
    )
    
    result = strategy.run()
    print(f"\n      Clustering complete!")
    print(f"      Number of clusters: {len(np.unique(result.labels[result.labels >= 0]))}")
    print(f"      Mean stability: {result.probabilities.mean():.3f}")
    
    # Save outputs
    output_files = strategy.save_outputs(save_to_model_specs=False)
    print(f"\n      Saved {len(output_files)} output files to: {b3_clustering_dir}")
    
    print(f"\n{'='*70}")
    print("✓ STEP 06b COMPLETE!")
    print(f"{'='*70}\n")


def b4(dr: bool = True, method: str = "k_precomputed"):
    # Get the paths
    cli_dir = Path(__file__).parent.parent / "cli"
    OUTDIR_B4 = OUTDIR_B4_DR if dr else OUTDIR_B4_NO_DR
    
    print(f"\n{'='*70}")
    print("SEMANTIC PCA PIPELINE - Bayley 4 (PAPER ITEMS ONLY)")
    print(f"{'='*70}")
    print(f"\nConfiguration (hardcoded for reproducibility):")
    print(f"  Model: {MODEL}")
    print(f"  Items file: {ITEMS_FILE_B4} (filtered to reference model items)")
    print(f"  Number of clusters: {N_CLUSTERS}")
    print(f"\nPipeline steps:")
    print(f"  Step 01 (Embeddings): {'SKIP' if SKIP_EMBED else 'RUN'}")
    print(f"  Step 03 (Clustering): {'SKIP' if SKIP_CLUSTER else 'RUN'}")
    print(f"  Step 04 (Labeling): {'SKIP' if SKIP_LABEL else 'RUN'}")
    
    # Step 01: Embeddings and Dimensionality Assessment
    if not SKIP_EMBED:
        cmd_01 = [
            sys.executable,
            str(cli_dir / "01_embed_and_assess.py"),
            "--model", MODEL,
            "--items-file", ITEMS_FILE_B4,
            "--output-dir", OUTDIR_B4,
        ]
        run_command(cmd_01, "Step 01: Embedding Generation & Dimensionality Assessment")
    else:
        print("\n⏭️  Skipping Step 01 (Embeddings)")
    
    # Step 03: Clustering Analysis
    if not SKIP_CLUSTER:
        if dr:
            cmd_03 = [
                sys.executable,
                str(cli_dir / "03_cluster_1000_kmean.py"),
                "--model", MODEL,
                "--n-clusters", str(N_CLUSTERS),
                "--strategy", STRATEGY,
                "--output-dir", OUTDIR_B4,
            ]
            run_command(cmd_03, "Step 03: UMAP + KMeans Clustering with Stability Analysis")
        else:
            cmd_03 = [
                sys.executable,
                str(cli_dir / "03_cluster_1000_kmean.py"),
                "--model", MODEL,
                "--n-clusters", str(N_CLUSTERS),
                "--strategy", STRATEGY,
                "--output-dir", OUTDIR_B4,
                "--no-dr",  # Disable dimensionality reduction
            ]
            run_command(cmd_03, "Step 03: UMAP + KMeans Clustering with Stability Analysis")
    else:
        print("\n⏭️  Skipping Step 03 (Clustering)")
    
    # Step 04: Cluster Labeling
    if not SKIP_LABEL:
        # Use embeddings from the output directory (run-specific)
        model_slug = MODEL.replace("/", "_").replace("-", "_")
        embeddings_path = f"{OUTDIR_B4}/embeddings_{model_slug}.npy"
        
        cmd_04 = [
            sys.executable,
            str(cli_dir / "04_labeling.py"),
            "--model", MODEL,
            "--embeddings", str(embeddings_path),
            "--labels-file", f"{OUTDIR_B4}/03_clustering/cluster_assignments.csv",
            "--no-def",
            "--output-dir", OUTDIR_B4,
        ]
        run_command(cmd_04, "Step 04: Cluster Centroid Labeling")
    else:
        print("\n⏭️  Skipping Step 04 (Labeling)")
    
    # Step 05: Model Comparison (B4 only - has reference models)
    if not SKIP_COMPARE:
        cmd_05 = [
            sys.executable,
            str(cli_dir / "05_compare_models.py"),
            "--model", MODEL,
            "--cluster-file", f"{OUTDIR_B4}/03_clustering/cluster_assignments.csv",
            "--cluster-labels-file", f"{OUTDIR_B4}/04_labeling/cluster_concept_labels.csv",
            "--output-dir", f"{OUTDIR_B4}/05_model_comparison",
        ]
        run_command(cmd_05, "Step 05: Model Comparison vs Reference Models")
    else:
        print("\n⏭️  Skipping Step 05 (Model Comparison)")
    
    # Step 06: Map b4 items to b3 items
    if not SKIP_BAYLEY_MAP:
        cmd_06 = [
            sys.executable,
            str(cli_dir / "06_b4_to_b3.py"),
            "--model", MODEL,
            "--items-file-b4", ITEMS_FILE_B4,
            "--items-file-b3", str(ITEMS_FILE_B3),
            "--output-dir", OUTDIR_B4,
        ]
        run_command(cmd_06, "Step 06: Map B4 Items to B3 Items")
    
    # Step 06b: Independent B3 Clustering on PAIRED items only (for fair comparison in Step 07)
    if not SKIP_BAYLEY_MAP:
        run_independent_b3_clustering(OUTDIR_B4, dr=dr)
    
    # Step 07: Evaluate b4-b3 Cluster Mapping (using independent B3 clustering)
    if not SKIP_BAYLEY_MAP:
        cmd_07 = [
            sys.executable,
            str(cli_dir / "07_compare_mapped_clusters.py"),
            "--model", MODEL,
            "--pairing-dir", OUTDIR_B4,
            "--b4-clustering", f"{OUTDIR_B4}/03_clustering/cluster_assignments.csv",
            "--b3-clustering", f"{OUTDIR_B4}/b3_independent_analysis/03_clustering/cluster_assignments.csv",
            "--output-dir", OUTDIR_B4,
        ]
        run_command(cmd_07, "Step 07: Compare B4 vs Independent B3 Clustering")


    # Extend from reference items to all B4 items (legacy; not used in the paper)
    if RUN_EXTENSION:
        extend_items(OUTDIR_B4, method=ASSIGNMENT_METHOD)
    else:
        print("\n⏭️  Skipping extension to all 81 B4 items (not used in the paper; --legacy-b4-b3 enables it)")

    # Final summary
    print(f"\n{'='*70}")
    print("🎉 B4 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    print("\nAll analysis steps finished. Check output directory for results:")
    print(f"  {OUTDIR_B4}")
    print()


def load_b4_clustering_data(outdir_b4: str):
    """Load the clustering results from the B4 analysis."""
    outdir = Path(outdir_b4)
    model_slug = MODEL.replace("/", "_").replace("-", "_")
    
    # Load original items and embeddings
    items_b4 = pd.read_csv(outdir / "items.csv")
    emb_b4 = np.load(outdir / f"embeddings_{model_slug}.npy")
    
    # Load cluster assignments and stability matrix
    cluster_assigns = pd.read_csv(outdir / "03_clustering" / "cluster_assignments.csv")
    
    # Load stability matrix (co-occurrence data)
    stability_summary = outdir / "03_clustering" / "stability_summary.json"
    
    return {
        'items_b4': items_b4,
        'emb_b4': emb_b4,
        'cluster_assigns': cluster_assigns,
        'stability_summary_path': stability_summary,
        'outdir': outdir,
        'model_slug': model_slug,
    }


def get_kmeans_consensus_centroids(clustering_data: dict) -> np.ndarray:
    """
    Extract centroids from the consensus clustering.
    
    For consensus clustering, we need to recompute centroids from the consensus labels.
    """
    items = clustering_data['items_b4']
    embeddings = clustering_data['emb_b4']
    cluster_assigns = clustering_data['cluster_assigns']
    
    n_clusters = len(cluster_assigns['cluster'].unique())
    n_features = embeddings.shape[1]
    centroids = np.zeros((n_clusters, n_features))
    
    for cluster_id in range(n_clusters):
        mask = cluster_assigns['cluster'] == cluster_id
        if mask.sum() > 0:
            centroids[cluster_id] = embeddings[mask].mean(axis=0)
    
    return centroids


def b4_extend_k_precomputed(outdir_b4: str):
    """
    Assign remaining B4 items to existing clusters using KMeans with precomputed centroids.
    
    This approach:
    1. Loads the clustering results from B4 analysis (paper_items_b4.txt)
    2. Extracts centroids from the consensus clustering
    3. Generates embeddings for remaining B4 items (b4_cog_items.txt)
    4. Runs KMeans on combined embeddings with init='precomputed' 
    5. Saves combined results with all 81 items
    """
    print(f"\n{'='*70}")
    print("EXTENDING B4 CLUSTERING - KMeans with Precomputed Centroids")
    print(f"{'='*70}\n")
        
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Load B4 clustering data
    print("[1/6] Loading B4 clustering results...")
    clustering_data = load_b4_clustering_data(outdir_b4)
    items_b4 = clustering_data['items_b4']
    emb_b4 = clustering_data['emb_b4']
    cluster_assigns = clustering_data['cluster_assigns']
    outdir = clustering_data['outdir']
    
    print(f"      Loaded {len(items_b4)} reference items")
    print(f"      Number of clusters: {cluster_assigns['cluster'].max() + 1}")
    
    # Extract centroids from consensus clustering
    print("\n[2/6] Extracting centroids from consensus clustering...")
    centroids = get_kmeans_consensus_centroids(clustering_data)
    n_clusters = centroids.shape[0]
    print(f"      Extracted {n_clusters} centroids (shape: {centroids.shape})")
    
    # Load remaining B4 items
    print("\n[3/6] Loading remaining B4 items...")
    items_all = parse_items(str(B4_ALL_ITEMS))
    items_remaining = items_all[~items_all['item_id'].isin(items_b4['item_id'])].copy()
    print(f"      Found {len(items_remaining)} remaining items")
    print(f"      Total items: {len(items_all)} ({len(items_b4)} reference + {len(items_remaining)} new)")
    
    # Generate embeddings for remaining items
    print("\n[4/6] Generating embeddings for remaining items...")
    model = load_model(MODEL)
    emb_remaining = embed_items(items_remaining, text_column="item_text", model=model)
    print(f"      Generated embeddings (shape: {emb_remaining.shape})")
    
    # Combine embeddings
    print("\n[5/6] Running KMeans with precomputed centroids on combined embeddings...")
    emb_combined = np.vstack([emb_b4, emb_remaining])
    print(f"      Combined embeddings shape: {emb_combined.shape}")
    
    # Run KMeans with precomputed centroids
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=centroids,
        n_init=1,
        random_state=42,
        max_iter=300,
    )
    labels_combined = kmeans.fit_predict(emb_combined)
    
    print(f"      KMeans converged in {kmeans.n_iter_} iterations")
    print(f"      Inertia: {kmeans.inertia_:.4f}")
    
    # Create combined assignments dataframe
    items_combined = pd.concat([items_b4, items_remaining], ignore_index=True)
    
    # For reference items, use original assignments
    cluster_assignments = pd.DataFrame({
        'item_id': items_combined['item_id'],
        'item_text': items_combined['item_text'],
        'cluster': labels_combined.astype(int),
        'is_reference': [i < len(items_b4) for i in range(len(items_combined))],
    })
    
    # For reference items, preserve original cluster assignment
    ref_clusters = cluster_assigns[['item_id', 'cluster']].copy()
    ref_mapping = dict(zip(ref_clusters['item_id'], ref_clusters['cluster']))
    cluster_assignments.loc[cluster_assignments['is_reference'], 'cluster'] = \
        cluster_assignments.loc[cluster_assignments['is_reference'], 'item_id'].map(ref_mapping).values
    
    # Create visualizations
    print(f"\n[6/7] Creating visualizations...")
    
    # Define output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_ext_dir = ensure_dir(str(outdir / "03_clustering_extended_kprecomputed"))
    
    # Visualization 1: Extended clustering (new items with red contours)
    reducer = get_reducer(
        "umap",
        verbose=False,
        n_components=2,
        random_state=42,
        n_neighbors=10,
        min_dist=0.01,
        metric="cosine"
    )
    X2d_ext = reducer.fit_transform(emb_combined)
    coords_ext = pd.DataFrame({
        'item_id': items_combined['item_id'],
        'umap_1': X2d_ext[:, 0],
        'umap_2': X2d_ext[:, 1]
    })
    create_extended_visualization(
        coords_2d=coords_ext,
        labels_combined=cluster_assignments['cluster'].values,
        is_reference=cluster_assignments['is_reference'].values,
        items_combined=items_combined,
        output_dir=output_ext_dir,
        dr_method="umap"
    )

    # Visualization 2: Cluster changes (items that moved clusters)
    coords_file = outdir / "03_clustering" / "umap_2d_coordinates.csv"
    if coords_file.exists():
        coords_ref = pd.read_csv(coords_file)
        coords_ref = coords_ref.merge(
            items_b4[['item_id', 'item_text']],
            on='item_id',
            how='left'
        )
        labels_original_ref = coords_ref['item_id'].map(ref_mapping).values
        labels_new_ref = coords_ref['item_id'].map(
            dict(zip(cluster_assignments['item_id'], cluster_assignments['cluster']))
        ).values
        create_cluster_change_visualization(
            coords_2d=coords_ref,
            labels_original=labels_original_ref,
            labels_new=labels_new_ref,
            items_df=coords_ref,
            output_dir=output_ext_dir,
            dr_method="umap"
        )
    else:
        print(f"      Warning: Could not find 2D coordinates file: {coords_file}")
        print("      Skipping cluster-change visualization.")
    
    print(f"\n[7/7] Saving results...")
    
    # Save combined assignments
    assignments_file = output_ext_dir / "cluster_assignments_combined.csv"
    cluster_assignments.to_csv(assignments_file, index=False)
    print(f"      Saved combined assignments: {assignments_file}")
    
    # Save combined embeddings
    emb_file = output_ext_dir / f"embeddings_{clustering_data['model_slug']}_combined.npy"
    np.save(emb_file, emb_combined)
    print(f"      Saved combined embeddings: {emb_file}")
    
    # Save combined items
    items_file = output_ext_dir / "items_combined.csv"
    items_combined.to_csv(items_file, index=False)
    print(f"      Saved combined items: {items_file}")
    
    # Save metadata
    metadata = {
        'method': 'kmeans_precomputed',
        'timestamp': timestamp,
        'n_items_reference': len(items_b4),
        'n_items_new': len(items_remaining),
        'n_items_total': len(items_combined),
        'n_clusters': n_clusters,
        'kmeans_inertia': float(kmeans.inertia_),
        'kmeans_n_iter': int(kmeans.n_iter_),
        'model': MODEL,
    }
    
    import json
    metadata_file = output_ext_dir / "extension_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"      Saved metadata: {metadata_file}")
    
    print(f"\n{'='*70}")
    print("✓ B4 EXTENSION (KMeans Precomputed) COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}\n")
    print(f"Summary:")
    print(f"  Reference items (paper_items_b4.txt): {len(items_b4)}")
    print(f"  New items (b4_cog_items.txt): {len(items_remaining)}")
    print(f"  Total items: {len(items_combined)}")
    print(f"  Clusters: {n_clusters}")
    print(f"\nOutput directory: {output_ext_dir}\n")


def b4_extend_nearest_centroid(outdir_b4: str):
    """
    Assign remaining B4 items to existing clusters using nearest centroid assignment.
    
    This approach:
    1. Loads the clustering results from B4 analysis (paper_items_b4.txt)
    2. Extracts centroids from the consensus clustering
    3. Generates embeddings for remaining B4 items (b4_cog_items.txt)
    4. Assigns each new item to the nearest centroid (cosine distance)
    5. Saves combined results with all 81 items
    """
    print(f"\n{'='*70}")
    print("EXTENDING B4 CLUSTERING - Nearest Centroid Assignment")
    print(f"{'='*70}\n")
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Load B4 clustering data
    print("[1/5] Loading B4 clustering results...")
    clustering_data = load_b4_clustering_data(outdir_b4)
    items_b4 = clustering_data['items_b4']
    emb_b4 = clustering_data['emb_b4']
    cluster_assigns = clustering_data['cluster_assigns']
    outdir = clustering_data['outdir']
    
    print(f"      Loaded {len(items_b4)} reference items")
    print(f"      Number of clusters: {cluster_assigns['cluster'].max() + 1}")
    
    # Extract centroids from consensus clustering
    print("\n[2/5] Extracting centroids from consensus clustering...")
    centroids = get_kmeans_consensus_centroids(clustering_data)
    n_clusters = centroids.shape[0]
    print(f"      Extracted {n_clusters} centroids (shape: {centroids.shape})")
    
    # Load remaining B4 items
    print("\n[3/5] Loading remaining B4 items...")
    items_all = parse_items(str(B4_ALL_ITEMS))
    items_remaining = items_all[~items_all['item_id'].isin(items_b4['item_id'])].copy()
    print(f"      Found {len(items_remaining)} remaining items")
    print(f"      Total items: {len(items_all)} ({len(items_b4)} reference + {len(items_remaining)} new)")
    
    # Generate embeddings for remaining items
    print("\n[4/5] Generating embeddings for remaining items and assigning...")
    model = load_model(MODEL)
    emb_remaining = embed_items(items_remaining, text_column="item_text", model=model)
    print(f"      Generated embeddings (shape: {emb_remaining.shape})")
    
    # Assign new items to nearest centroid
    distances = cosine_distances(emb_remaining, centroids)
    labels_remaining = distances.argmin(axis=1)
    nearest_distances = distances.min(axis=1)
    
    print(f"      Assigned {len(items_remaining)} new items to nearest centroids")
    print(f"      Mean cosine distance to nearest centroid: {nearest_distances.mean():.4f}")
    print(f"      Std cosine distance to nearest centroid: {nearest_distances.std():.4f}")
    
    # Create combined assignments dataframe
    items_combined = pd.concat([items_b4, items_remaining], ignore_index=True)
    
    # Reference items keep original assignments
    labels_combined = np.concatenate([
        cluster_assigns['cluster'].values,
        labels_remaining
    ])
    
    cluster_assignments = pd.DataFrame({
        'item_id': items_combined['item_id'],
        'item_text': items_combined['item_text'],
        'cluster': labels_combined,
        'is_reference': [i < len(items_b4) for i in range(len(items_combined))],
        'distance_to_centroid': np.concatenate([
            np.zeros(len(items_b4)),  # Reference items have no distance metric
            nearest_distances
        ])
    })

    # Create visualizations
    print(f"\n[5/6] Creating visualizations...")
    
    # Define output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_ext_dir = ensure_dir(str(outdir / "03_clustering_extended_nearest_centroid"))
    
    # Visualization 1: Extended clustering (new items with red contours)
    emb_combined = np.vstack([emb_b4, emb_remaining])
    reducer = get_reducer(
        "umap",
        verbose=False,
        n_components=2,
        random_state=42,
        n_neighbors=10,
        min_dist=0.01,
        metric="cosine"
    )
    X2d_ext = reducer.fit_transform(emb_combined)
    coords_ext = pd.DataFrame({
        'item_id': items_combined['item_id'],
        'umap_1': X2d_ext[:, 0],
        'umap_2': X2d_ext[:, 1]
    })
    create_extended_visualization(
        coords_2d=coords_ext,
        labels_combined=labels_combined,
        is_reference=np.array([i < len(items_b4) for i in range(len(items_combined))]),
        items_combined=items_combined,
        output_dir=output_ext_dir,
        dr_method="umap"
    )
    
    print(f"\n[6/6] Saving results...")
    
    # Save combined assignments
    assignments_file = output_ext_dir / "cluster_assignments_combined.csv"
    cluster_assignments.to_csv(assignments_file, index=False)
    print(f"      Saved combined assignments: {assignments_file}")
    
    # Save combined embeddings
    emb_combined = np.vstack([emb_b4, emb_remaining])
    emb_file = output_ext_dir / f"embeddings_{clustering_data['model_slug']}_combined.npy"
    np.save(emb_file, emb_combined)
    print(f"      Saved combined embeddings: {emb_file}")
    
    # Save combined items
    items_file = output_ext_dir / "items_combined.csv"
    items_combined.to_csv(items_file, index=False)
    print(f"      Saved combined items: {items_file}")
    
    # Save metadata
    metadata = {
        'method': 'nearest_centroid',
        'timestamp': timestamp,
        'n_items_reference': len(items_b4),
        'n_items_new': len(items_remaining),
        'n_items_total': len(items_combined),
        'n_clusters': n_clusters,
        'mean_distance_to_centroid': float(nearest_distances.mean()),
        'std_distance_to_centroid': float(nearest_distances.std()),
        'model': MODEL,
    }
    
    import json
    metadata_file = output_ext_dir / "extension_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"      Saved metadata: {metadata_file}")
    
    print(f"\n{'='*70}")
    print("✓ B4 EXTENSION (Nearest Centroid) COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}\n")
    print(f"Summary:")
    print(f"  Reference items (paper_items_b4.txt): {len(items_b4)}")
    print(f"  New items (b4_cog_items.txt): {len(items_remaining)}")
    print(f"  Total items: {len(items_combined)}")
    print(f"  Clusters: {n_clusters}")
    print(f"  Mean distance to nearest centroid: {nearest_distances.mean():.4f}")
    print(f"\nOutput directory: {output_ext_dir}\n")


def extend_items(outdir_b4: str, method: str = "k_precomputed"):
    print(f"\n{'='*70}")
    print("EXTENDING CLUSTERING TO ALL B4 ITEMS")
    print(f"{'='*70}\n")
    
    if method == "k_precomputed":
        b4_extend_k_precomputed(outdir_b4)
    elif method == "nearest_centroid":
        b4_extend_nearest_centroid(outdir_b4)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Objective 2: consensus K-Means validation on the 39 Bayley-4 items.")
    ap.add_argument("--from-cache", action="store_true",
                    help="skip embedding (01), clustering (03) and labelling (04); "
                         "recompute only the reference-model comparison (05) on the "
                         "committed run directory")
    ap.add_argument("--with-dr", action="store_true",
                    help="also run the variant with UMAP dimensionality reduction before "
                         "every K-Means run (exploratory; not used in the paper) and build "
                         "the DR vs no-DR comparison workbook")
    ap.add_argument("--legacy-b4-b3", action="store_true",
                    help="also run the legacy greedy Bayley-4 -> Bayley-III mapping "
                         "(06/06b/07) and the extension to all 81 Bayley-4 items "
                         "(superseded by the cascade pairing in step 1; not used in the paper)")
    return ap.parse_args(argv)


def main(argv=None):
    global SKIP_EMBED, SKIP_CLUSTER, SKIP_LABEL, SKIP_BAYLEY_MAP, RUN_EXTENSION
    args = parse_args(argv)
    if args.from_cache:
        SKIP_EMBED = SKIP_CLUSTER = SKIP_LABEL = True
    if args.legacy_b4_b3:
        SKIP_BAYLEY_MAP = False
        RUN_EXTENSION = True

    # Paper configuration: no dimensionality reduction before clustering.
    b4(dr=False, method=ASSIGNMENT_METHOD)

    if args.with_dr:
        b4(dr=True, method=ASSIGNMENT_METHOD)
        # Summary Excel comparing the B4 runs with and without DR
        cmd_summary = [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "cli" / "generate_comparison_excel.py"),
            "--comparison",
            "--folder-path", str(outpath),
        ]
        run_command(cmd_summary, "Final Step: Summary Excel Generation")


if __name__ == "__main__":
    main()
