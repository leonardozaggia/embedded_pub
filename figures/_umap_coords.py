"""
_umap_coords.py — one-off UMAP embedding precompute (deterministic cache).

Runs UMAP once with the project's best parameters and writes the 3-D and 2-D
coordinates plus the theoretical/detected factor labels to
data/outputs/figure_inputs/umap3d_coords.csv. The figure scripts then read this
cache instead of re-running UMAP, so panel B of Figure 2 is exactly reproducible
and fast to re-render.

Reuses _umap_helpers.py::load_data() (extracted from the original exploratory
umap3d_beautiful.py) so the geometry is identical to the existing project figure.
The committed cache is the one used for the manuscript figure; re-running this
script with a different umap-learn / numba version can move points slightly.

The panel-B *layout* (cluster separation / spread) is controlled HERE, not in
fig2_embedding.py — after changing it you must re-run this script to rebuild the
cache, then re-run fig2_embedding.py.

Usage:
    python figures/_umap_coords.py                       # defaults below
    python figures/_umap_coords.py --target-weight 0.30 --min-dist 0.40
      --target-weight  cluster separation (higher = tighter/more separated)
      --min-dist       within-cluster spread (higher = more scattered points)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import umap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import _umap_helpers as ub      # noqa: E402  (load_data + BEST_* params)
import _embedding as emb        # noqa: E402  (translated_short_map)

CACHE = ROOT / emb.UMAP_CACHE

# UNSUPERVISED layout (target_weight=0): the honest embedding geometry, coloured by
# the translated model. The translated domains partly overlap by nature (WM/GDPS/HOP
# proximity), so fig2 draws POINTS ONLY (no hulls) — see fig2_embedding.py. `spread`
# separates the natural structure on a larger canvas; `min_dist` keeps points tight;
# `n_neighbors` favours global structure. Set target_weight>0 for a forced (circular)
# separation. Panel C is unaffected — it uses raw cosine similarity, not these coords.
N_NEIGHBORS = 30
SPREAD = 2.5
MIN_DIST = 0.10
TARGET_WEIGHT = 0.0


def _reducer(n_components: int, seed: int, target_weight: float = TARGET_WEIGHT,
             min_dist: float = MIN_DIST, spread: float = SPREAD,
             n_neighbors: int = N_NEIGHBORS):
    return umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        metric="cosine",
        n_epochs=300,
        random_state=seed,
        low_memory=False,
        target_weight=target_weight,
    )


def _translated_targets(item_ids) -> np.ndarray:
    """Integer-encode each item's translated factor (mixed_centroids, UC→best_ef)
    for semi-supervised UMAP; unmapped items → -1 (unlabelled)."""
    tmap = emb.translated_short_map(ROOT)
    labels = [tmap.get(iid) for iid in item_ids]
    classes = sorted({l for l in labels if l})
    cmap = {c: i for i, c in enumerate(classes)}
    return np.array([cmap.get(l, -1) for l in labels], dtype=int)


def compute(seed: int = 42, target_weight: float = TARGET_WEIGHT,
            min_dist: float = MIN_DIST, spread: float = SPREAD,
            n_neighbors: int = N_NEIGHBORS) -> pd.DataFrame:
    df, embeddings = ub.load_data()
    # y only used when semi-supervising (target_weight > 0); unsupervised by default.
    y = _translated_targets(list(df["item_id"])) if target_weight > 0 else None
    r3 = _reducer(3, seed, target_weight, min_dist, spread, n_neighbors)
    r2 = _reducer(2, seed, target_weight, min_dist, spread, n_neighbors)
    coords3 = r3.fit_transform(embeddings, y=y)
    coords2 = r2.fit_transform(embeddings, y=y)   # 2-D variant
    df = df.copy()
    df["x"], df["y"], df["z"] = coords3[:, 0], coords3[:, 1], coords3[:, 2]
    df["x2"], df["y2"] = coords2[:, 0], coords2[:, 1]
    return df


def main():
    ap = argparse.ArgumentParser(description="Precompute the fig2 UMAP cache.")
    ap.add_argument("--n-neighbors", type=int, default=N_NEIGHBORS,
                    help="local vs global structure")
    ap.add_argument("--spread", type=float, default=SPREAD,
                    help="cluster separation lever (higher = more separated)")
    ap.add_argument("--min-dist", type=float, default=MIN_DIST,
                    help="within-cluster spread (higher = more scattered)")
    ap.add_argument("--target-weight", type=float, default=TARGET_WEIGHT,
                    help="0 = unsupervised (natural); >0 forces label separation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print(f"Loading embeddings + labels … (n_neighbors={args.n_neighbors}, "
          f"spread={args.spread}, min_dist={args.min_dist}, "
          f"target_weight={args.target_weight}, seed={args.seed})")
    df = compute(seed=args.seed, target_weight=args.target_weight, min_dist=args.min_dist,
                 spread=args.spread, n_neighbors=args.n_neighbors)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    # item_text is deliberately NOT written: the item wording is proprietary
    # (see docs/DATA_ACCESS.md); the cache only needs ids, labels and coordinates.
    cols = [c for c in ["item_id", "detected", "theoretical",
                        "x", "y", "z", "x2", "y2"] if c in df.columns]
    df[cols].to_csv(CACHE, index=False)
    print(f"Cached {len(df)} item coordinates → {CACHE}")
    print("  theoretical counts:", df["theoretical"].value_counts().to_dict())
    print("  detected counts   :", df["detected"].value_counts().to_dict())


if __name__ == "__main__":
    main()
