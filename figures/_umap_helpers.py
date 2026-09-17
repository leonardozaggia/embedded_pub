"""
_umap_helpers.py -- shared helpers for the UMAP panels of Figure 2.

Verbatim extraction of the parts of the original exploratory script
``figures/umap3d_beautiful.py`` (3-D UMAP with ellipsoidal cluster hulls) that
the publication figures depend on: the locked palette/markers, the UMAP
parameters found in the permutation sweep, the wireframe/ellipsoid drawing
helpers and ``load_data()``, which assembles item ids, detected (bottom-up)
clusters and translated factors from the committed pipeline outputs.

Nothing here is meant to be run directly; ``_umap_coords.py`` and
``_embedding.py`` import it.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]   # figures/ -> repository root
STEP3_RUN_DIR = (
    REPO_ROOT / "data" / "outputs" / "step3_b3_bottomup"
    / "all_mpnet_base_v2_kmeans_consensus_no_dr"
)
STEP1_BASE  = REPO_ROOT / "data" / "outputs" / "step1_translation"

# ── Best UMAP parameters (from permutation sweep) ─────────────────────────────
BEST_N_NEIGHBORS = 20
BEST_MIN_DIST    = 0.0
BEST_SPREAD      = 2.0
BEST_METRIC      = "cosine"
BEST_N_EPOCHS    = 200

# ── Refined, muted colour palette ─────────────────────────────────────────────
# Desaturated relative to the previous vivid palette; matches the cool
# professional hues in the Wulff & Mata figure while still being
# distinguishable at small size and in greyscale.
COLORS = {
    "WM":   "#2D5FA3",   # royal blue
    "FS":   "#D4783A",   # warm amber
    "GDPS": "#2E9E8A",   # teal
    "ATT":  "#B5404E",   # brick rose
    "HOP":  "#7755A8",   # soft violet
}
# Cluster-hull colours are slightly lighter / more transparent variants
HULL_COLORS = {k: v for k, v in COLORS.items()}   # same hue, alpha handled per-call

MARKERS = {
    "WM":   "o",
    "FS":   "s",
    "GDPS": "^",
    "ATT":  "D",
    "HOP":  "P",
}
FACTOR_ORDER = ["WM", "FS", "GDPS", "ATT", "HOP"]

DETECTED_NAME_MAP = {
    "working memory":               "WM",
    "attention":                    "ATT",
    "higher order processes":       "HOP",
    "higher order processing":      "HOP",
    "flexibility shift":            "FS",
    "goal directed problem solving":"GDPS",
}
THEO_NAME_MAP = {
    "working_memory": "WM", "working memory": "WM",
    "flexibility_shift": "FS", "flexibility shift": "FS",
    "goal_directed_problem_solving": "GDPS", "goal directed problem solving": "GDPS",
    "attention": "ATT",
    "higher_order_processing": "HOP", "higher order processing": "HOP",
    "higher order processes": "HOP",
    "uncategorized_cognition": "Unclassified",
}


# ── Geometry helpers ──────────────────────────────────────────────────────────
def _draw_wireframe_box(ax, x_lim, y_lim, z_lim,
                        color: str = "#BBBBBB", lw: float = 0.7, alpha: float = 0.55):
    """Draw the 12 edges of a rectangular box — no filled faces."""
    x0, x1 = x_lim
    y0, y1 = y_lim
    z0, z1 = z_lim
    edges = [
        # bottom face
        ([x0,x1],[y0,y0],[z0,z0]),  ([x0,x0],[y0,y1],[z0,z0]),
        ([x1,x1],[y0,y1],[z0,z0]),  ([x0,x1],[y1,y1],[z0,z0]),
        # top face
        ([x0,x1],[y0,y0],[z1,z1]),  ([x0,x0],[y0,y1],[z1,z1]),
        ([x1,x1],[y0,y1],[z1,z1]),  ([x0,x1],[y1,y1],[z1,z1]),
        # vertical edges
        ([x0,x0],[y0,y0],[z0,z1]),  ([x1,x1],[y0,y0],[z0,z1]),
        ([x1,x1],[y1,y1],[z0,z1]),  ([x0,x0],[y1,y1],[z0,z1]),
    ]
    for xe, ye, ze in edges:
        ax.plot(xe, ye, ze, color=color, lw=lw, alpha=alpha, zorder=1)


def _draw_ellipsoid(ax, center: np.ndarray, cov: np.ndarray,
                    color: str, n_std: float = 1.8,
                    alpha: float = 0.13, n_pts: int = 45):
    """Fit a multivariate Gaussian, draw its n_std-sigma ellipsoidal surface."""
    try:
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-10)
        # Regularise: prevent degenerate flat/needle ellipsoids for small or
        # nearly-collinear clusters (e.g. HOP with n=6 points).  Clamp each
        # eigenvalue to at least 15% of the largest so the hull stays legibly 3D.
        vals = np.maximum(vals, 0.15 * vals.max())

        phi   = np.linspace(0, 2 * np.pi, n_pts)
        theta = np.linspace(0, np.pi,     n_pts)
        sp = np.outer(np.cos(phi), np.sin(theta))
        ss = np.outer(np.sin(phi), np.sin(theta))
        sc = np.outer(np.ones_like(phi), np.cos(theta))

        sphere  = np.stack([sp, ss, sc], axis=0)          # (3, n, n)
        scaled  = (n_std * np.sqrt(vals))[:, None, None] * sphere
        rotated = np.einsum("ij,jkl->ikl", vecs, scaled)

        xe = center[0] + rotated[0]
        ye = center[1] + rotated[1]
        ze = center[2] + rotated[2]

        ax.plot_surface(xe, ye, ze,
                        alpha=alpha, color=color,
                        linewidth=0, shade=True,
                        antialiased=True, zorder=2)
    except Exception:
        pass


# ── Data loading ──────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    return s.strip().lower().replace("-", " ").replace("_", " ")


def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    embeddings  = np.load(STEP3_RUN_DIR / "embeddings_all_mpnet_base_v2.npy")
    items_df    = pd.read_csv(STEP3_RUN_DIR / "items.csv")
    cluster_df  = pd.read_csv(STEP3_RUN_DIR / "03_clustering" / "cluster_assignments.csv")
    cl_labels   = pd.read_csv(STEP3_RUN_DIR / "04_labeling"  / "cluster_concept_labels.csv")

    df = items_df[["item_id"]].copy()
    df["item_text"] = (
        items_df.get("item_text", items_df.get("title", items_df["item_id"]))
    )
    df = df.merge(cluster_df[["item_id", "cluster"]], on="item_id", how="left")

    cid_to_name = {
        int(r["cluster_id"]): DETECTED_NAME_MAP.get(_norm(str(r["top_concept"])),
                                                      str(r["top_concept"]).upper()[:5])
        for _, r in cl_labels.iterrows()
    }
    df["detected"] = df["cluster"].map(cid_to_name).fillna("?")

    # ── Theoretical factor from step1 ────────────────────────────────────────
    item_to_factor: dict[str, str] = {}
    for src in [
        STEP1_BASE / "mixed_centroids" / "all_unpaired_b3_assignments.csv",
        STEP1_BASE / "b3_centroids"    / "all_unpaired_b3_assignments.csv",
    ]:
        if src.exists():
            u = pd.read_csv(src)
            col = "assigned_factor" if "assigned_factor" in u.columns else (
                  "factor"           if "factor"           in u.columns else None)
            if col:
                for _, row in u.iterrows():
                    iid = str(row["item_id"])
                    if iid not in item_to_factor:
                        item_to_factor[iid] = THEO_NAME_MAP.get(
                            _norm(str(row[col])), _norm(str(row[col])).upper()[:5])
            break

    paired_csv = STEP1_BASE / "b3_paired_assignments.csv"
    if paired_csv.exists():
        p   = pd.read_csv(paired_csv)
        col = "assigned_factor" if "assigned_factor" in p.columns else (
              "factor"           if "factor"           in p.columns else None)
        if col:
            for _, row in p.iterrows():
                iid = str(row["item_id"])
                if iid not in item_to_factor:
                    item_to_factor[iid] = THEO_NAME_MAP.get(
                        _norm(str(row[col])), _norm(str(row[col])).upper()[:5])

    df["theoretical"] = df["item_id"].map(item_to_factor).fillna("Unclassified")
    return df, embeddings


