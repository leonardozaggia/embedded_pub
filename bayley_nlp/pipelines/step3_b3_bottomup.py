#!/usr/bin/env python
"""
Objective 3 -- Bottom-up clustering of all 91 Bayley-III cognitive items.

The same unsupervised pipeline as Objective 2 (embed -> consensus K-Means with
1,000 runs -> post-hoc labelling by nearest domain name) is applied to the
full Bayley-III cognitive scale (COG_001 to COG_091).  The resulting solution
is compared with the translated structure from Objective 1 by
``analysis/concordance/b3_*.py`` and ``figures/fig5_validation.py``.

The manuscript reports the run WITHOUT dimensionality reduction before
clustering (output folder ``all_mpnet_base_v2_kmeans_consensus_no_dr``).

Usage:
    python -m bayley_nlp.pipelines.step3_b3_bottomup               # paper configuration
    python -m bayley_nlp.pipelines.step3_b3_bottomup --from-cache  # re-label only (skip 01/03)
    python -m bayley_nlp.pipelines.step3_b3_bottomup --with-dr     # UMAP-before-K-Means variant
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
import re
import subprocess
import sys
from pathlib import Path
import os

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from bayley_nlp.core import parse_items


# ============================================================================
# CONFIGURATION - Edit these values to change pipeline settings
# ============================================================================

MODEL = "all-mpnet-base-v2"  # Options: "all-mpnet-base-v2", "all-roberta-large-v1"
ITEMS_FILE_B3 = repo_root / "data/raw/items_description/b3_cog_items.txt"
ITEM_START = 1
ITEM_END = 91
STRATEGY = "kmeans_consensus"  # ["kmeans_consensus", "kmeans_simple", "hdbscan", "kmeans_multi_init", "similarity"]
N_CLUSTERS = 5

# Skip flags - set to True to skip steps when re-running parts.
SKIP_EMBED = False
SKIP_CLUSTER = False
SKIP_LABEL = False

# Directories (anchored at the repository root so the script can be run from
# any working directory)
outpath = repo_root / "data/outputs/step3_b3_bottomup"
OUTDIR_B3_DR = outpath / f"{MODEL.replace('-', '_')}_{STRATEGY}_dr"
OUTDIR_B3_NO_DR = outpath / f"{MODEL.replace('-', '_')}_{STRATEGY}_no_dr"
# ============================================================================


def run_command(cmd: list[str], step_name: str) -> subprocess.CompletedProcess:
    """Run a command and stop the pipeline if it fails."""
    print(f"\n{'=' * 70}")
    print(f"RUNNING: {step_name}")
    print(f"{'=' * 70}")
    print(f"Command: {' '.join(cmd)}\n")

    # Ensure subprocesses can import the top-level `bayley_nlp` package by
    # adding the repository root to PYTHONPATH. When Python runs a script by
    # filename, sys.path[0] is the script's directory (bayley_nlp/cli), which
    # prevents sibling package imports from resolving. Adding the repo root
    # ensures `import bayley_nlp` works in the spawned process.
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[2]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + existing if existing else "")

    result = subprocess.run(cmd, capture_output=False, text=True, env=env)

    if result.returncode != 0:
        print(f"\nERROR: {step_name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n{step_name} completed successfully.")
    return result


def item_num_from_id(item_id: str) -> int:
    match = re.search(r"(\d+)$", item_id)
    if match is None:
        raise ValueError(f"Could not parse numeric item id from {item_id!r}")
    return int(match.group(1))


def validate_b3_items_file() -> Path:
    """Validate that the Bayley-III source file contains the full item set."""
    source_path = Path(ITEMS_FILE_B3)

    items = parse_items(str(source_path))
    item_numbers = items["item_id"].map(item_num_from_id)

    expected_n = ITEM_END - ITEM_START + 1
    expected_item_numbers = list(range(ITEM_START, ITEM_END + 1))
    actual_item_numbers = item_numbers.tolist()
    if actual_item_numbers != expected_item_numbers:
        raise ValueError(
            f"Expected {expected_n} Bayley-III items from COG_{ITEM_START:03d} "
            f"through COG_{ITEM_END:03d}, but found {len(items)} items with "
            f"range COG_{actual_item_numbers[0]:03d}-COG_{actual_item_numbers[-1]:03d}."
        )

    print(f"Validated full Bayley-III item file: {source_path} ({len(items)} items)")
    return source_path


def run_b3_pipeline(dr: bool = True) -> None:
    cli_dir = Path(__file__).parent.parent / "cli"
    outdir = OUTDIR_B3_DR if dr else OUTDIR_B3_NO_DR
    items_file = validate_b3_items_file()

    print(f"\n{'=' * 70}")
    print("SEMANTIC PCA PIPELINE - Bayley-III FULL ITEM SET")
    print(f"{'=' * 70}")
    print("\nConfiguration:")
    print(f"  Model: {MODEL}")
    print(f"  Items file: {items_file}")
    print(f"  Item set: COG_{ITEM_START:03d}-COG_{ITEM_END:03d}")
    print(f"  Number of clusters: {N_CLUSTERS}")
    print(f"  Dimensionality reduction: {'Yes' if dr else 'No'}")
    print("\nPipeline steps:")
    print(f"  Step 01 (Embeddings): {'SKIP' if SKIP_EMBED else 'RUN'}")
    print(f"  Step 03 (Clustering): {'SKIP' if SKIP_CLUSTER else 'RUN'}")
    print(f"  Step 04 (Labeling): {'SKIP' if SKIP_LABEL else 'RUN'}")

    if not SKIP_EMBED:
        cmd_01 = [
            sys.executable,
            str(cli_dir / "01_embed_and_assess.py"),
            "--model",
            MODEL,
            "--items-file",
            items_file.name,
            "--output-dir",
            str(outdir),
        ]
        run_command(cmd_01, "Step 01: Bayley-III Embedding Generation & Dimensionality Assessment")
    else:
        print("\nSkipping Step 01 (Embeddings)")

    if not SKIP_CLUSTER:
        cmd_03 = [
            sys.executable,
            str(cli_dir / "03_cluster_1000_kmean.py"),
            "--model",
            MODEL,
            "--n-clusters",
            str(N_CLUSTERS),
            "--strategy",
            STRATEGY,
            "--output-dir",
            str(outdir),
        ]
        if not dr:
            cmd_03.append("--no-dr")
        run_command(cmd_03, "Step 03: Bayley-III KMeans Clustering with Stability Analysis")
    else:
        print("\nSkipping Step 03 (Clustering)")

    if not SKIP_LABEL:
        model_slug = MODEL.replace("/", "_").replace("-", "_")
        embeddings_path = outdir / f"embeddings_{model_slug}.npy"
        cmd_04 = [
            sys.executable,
            str(cli_dir / "04_labeling.py"),
            "--model",
            MODEL,
            "--embeddings",
            str(embeddings_path),
            "--labels-file",
            str(outdir / "03_clustering" / "cluster_assignments.csv"),
            "--items",
            str(outdir / "items.csv"),
            "--no-def",
            "--output-dir",
            str(outdir),
        ]
        run_command(cmd_04, "Step 04: Bayley-III Cluster Centroid Labeling")
    else:
        print("\nSkipping Step 04 (Labeling)")

    print(f"\n{'=' * 70}")
    print("BAYLEY-III PIPELINE COMPLETED")
    print(f"{'=' * 70}")
    print(f"Output directory: {outdir}\n")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Objective 3: consensus K-Means on all 91 Bayley-III items.")
    ap.add_argument("--from-cache", action="store_true",
                    help="skip embedding (01) and clustering (03); re-run only the "
                         "post-hoc labelling (04) on the committed run directory")
    ap.add_argument("--with-dr", action="store_true",
                    help="run the variant with UMAP dimensionality reduction before every "
                         "K-Means run instead of the paper configuration (exploratory)")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    global SKIP_EMBED, SKIP_CLUSTER
    args = parse_args(argv)
    if args.from_cache:
        SKIP_EMBED = SKIP_CLUSTER = True
    run_b3_pipeline(dr=args.with_dr)


if __name__ == "__main__":
    main()
