#!/usr/bin/env python
"""
Objective 1 -- expert-guided translation of the Bayley-4 domain structure
onto the Bayley-III (step 1 of the pipeline; `python -m bayley_nlp step1`).

Embeds the Bayley-4 items assigned by Aylward et al. (2022) and all 91
Bayley-III items with all-mpnet-base-v2, pairs each Bayley-4 item with a
Bayley-III item by cascade matching (title cosine >= 0.75 locks a pair,
otherwise full-text similarity; greedy and exclusive), transfers the five
domain labels, assigns the unpaired Bayley-III items to the nearest domain
centroid (with a leave-one-out unclassified cluster, UC; items whose UC
similarity exceeds the best domain by > 0.10 are flagged for expert review),
and writes the translated domains of items 34-68 into config/model_specs.yaml
for the Mplus measurement model.  The expert review of the flagged items and
the final 91-item taxonomy are handled afterwards by
expert_review/import_review_b3.py and analysis/item_assignments.py.

Outputs: data/outputs/step1_translation/  (b4_b3_cascade_pairs.csv = Table S1,
b3_paired_assignments.csv, <method>/all_unpaired_b3_assignments.csv,
<method>/b3_model_spec.yaml, diagnostic figures incl. the centroid heatmaps
behind Figures S1-S2).

Details
-------
The model-spec keys, figure titles and Mplus input files tag this assignment
scheme "v6_cog".

1. UC-assigned items are reassigned to their best EF cognitive facet
   Items whose LOO centroid assignment is UC (uncategorized cognition)
   are redirected to the EF cognitive facet with the highest cosine
   similarity among all expert-guided facet centroids.  The assign-
   ment to UC is the LOO rule's first choice; the *model spec*
   uses the second choice (best EF facet) so that every item in
   the 34-68 dHCP set is placed in a theoretically motivated facet.

2. Sparse cognitive facets are excluded (not merged to UC)
   Cognitive facets with < MIN_ITEMS_PER_EF_FACTOR items in the
   34-68 dHCP set are dropped entirely: their items are excluded
   from the MIRT model rather than merged into a residual factor.
   This keeps the model parsimonious and interpretable.

   Consequence: no UC factor in the model spec.  ATT and HOP
   (each with only 1 paired item in 34-68) are always excluded.

3. Columns of the LOO output
   assign_via_centroids_loo() records:
     best_ef_factor   — top EF facet (ignoring UC)
     best_ef_sim      — its cosine similarity
     second_ef_factor — 2nd-best EF facet
     second_ef_sim    — its cosine similarity
     uc_margin        — sim_UC − best_ef_sim
                        (positive ⇒ UC was genuinely closest)
   These columns drive figures 01c, 01d and 07 and the UC
   similarity profile diagnostics.

4. Figures
   01c — All 91 B3 items (mixed centroids, dHCP highlighted): UC-redirected
         dHCP items linked to their best EF facet centroid; items where
         uc_margin > UC_UNCERTAINTY_MARGIN (UC strongly dominates) flagged
         with hollow marker and "?" annotation.
   07  — Two sub-panels: lollipop (per-item all-facet sims for UC
         redirected items) + scatter (sim_UC vs sim_best_ef for all
         LOO items). Red = strong UC preference (margin > threshold).

Pipeline
--------
1.  Load expert-guided B4 cognitive facet assignments.
2.  Embed B4 (restricted to expert-guided) and all 91 B3 items.
3.  Cascade B4↔B3 pairing: title-locked (sim_to >= 0.75) then full-text greedy.
4.  Compute visual UC centroid (mean of ALL unpaired B3; display anchor).
5.  For each of three EF centroid methods (b4_centroids, b3_centroids,
    mixed_centroids):
      a. Compute 5 cognitive facet centroids (cross-loader-aware).
      b. Run LOO centroid assignment on ALL unpaired B3 items.
         Records best_ef_factor, second_ef_factor, uc_margin per item.
      c. Save all_unpaired_b3_assignments.csv; filter to 34-68.
      d. Redirect LOO-UC items to best_ef_factor in model spec.
      e. Identify and exclude sparse cognitive facets.
      f. Save model spec (dynamic F-numbering, no UC factor).
      g. Compute and save UC similarity profile (diagnostics).
6.  Generate all figures including 01c, 01d, and 07.

Output layout
-------------
data/outputs/step1_translation/
  b4_b3_greedy_pairs.csv
  b3_paired_assignments.csv
  b4_unpaired_items.csv
  b3_unpaired_items.csv
  b4_centroids/
    all_unpaired_b3_assignments.csv
    unpaired_34_68_assignments.csv
    full_34_68_assignments.csv
    uc_similarity_profile.csv
    b3_model_spec.yaml
  b3_centroids/   ...
  mixed_centroids/ ...
  figures/
    01_pairing_mds.png
    01b_pairing_three_subplots.png
    01c_b3_dhcp_uc_redirected.png
    01d_all_b3_uc_redirected.png
    02_b3_factor_assignment_umap.png
    02b_b3_factor_dr_variants.png
    03_centroid_sim_heatmap_{method}.png
    04_centroid_assignment_comparison.png
    05_factor_sizes.png
    06_uc_similarity_profile.png
    07_uc_decision_logic.png

Usage
-----
    python -m bayley_nlp.pipelines.step1_translation
"""

from __future__ import annotations

# --- UTF-8 stdout/stderr: this script prints Unicode (arrows, checkmarks)
# which crashes under Windows' default cp1252 encoding; force UTF-8. ---
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")
del _stream


import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Repository root so the script can be run from any working directory.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bayley_nlp.core import embed_items, load_model, parse_items
from bayley_nlp.core.dimensionality_reduction import get_reducer

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL = "all-mpnet-base-v2"

ITEMS_FILE_B4 = _REPO_ROOT / "data/raw/items_description/paper_items_b4.txt"
ITEMS_FILE_B3_ALL = _REPO_ROOT / "data/raw/items_description/b3_cog_items.txt"
ITEMS_FILE_B3_DATA = _REPO_ROOT / "data/raw/items_description/b3_cog_items_34_68.txt"
REFERENCE_MODELS_FILE = _REPO_ROOT / "config/reference_models_b4.yaml"
MODEL_SPECS_FILE = _REPO_ROOT / "config/model_specs.yaml"

OUTDIR = _REPO_ROOT / "data/outputs/step1_translation"

# Label used throughout for the uncategorized cognition residual factor
UNNAMED_CLUSTER = "uncategorized_cognition"

# Minimum number of items a cognitive facet must have in the 34-68 dHCP set
# to be included in the MIRT model spec. Facets below this threshold are
# excluded entirely (their items are dropped from the model, not reassigned).
MIN_ITEMS_PER_EF_FACTOR = 3

# Uncertainty threshold for 01c/01d: if uc_margin < this, flag as uncertain
UC_UNCERTAINTY_MARGIN = 0.10
# Tie threshold: if |sim_rank2_ef - sim_rank3_ef| < this, flag as tie-uncertain
TIE_THRESHOLD = 0.01

# Canonical factor order (defines F1 … F5 mapping — UC added dynamically)
FACTOR_ORDER = [
    "attention",
    "working_memory",
    "goal_directed_problem_solving",
    "flexibility_shift",
    "higher_order_processing",
]

FACTOR_ABBREV: dict[str, str] = {
    "attention":                     "ATT",
    "working_memory":                "WM",
    "goal_directed_problem_solving": "GDPS",
    "flexibility_shift":             "FS",
    "higher_order_processing":       "HOP",
    UNNAMED_CLUSTER:                 "UC",
}

FACTOR_COLORS: dict[str, str] = {
    "attention":                     "#C44E52",
    "working_memory":                "#4C72B0",
    "goal_directed_problem_solving": "#55A868",
    "flexibility_shift":             "#DD8452",
    "higher_order_processing":       "#8172B3",
    UNNAMED_CLUSTER:                 "#888888",
    "unassigned":                    "#AAAAAA",
}

CENTROID_METHODS = ["b4_centroids", "b3_centroids", "mixed_centroids"]

# ============================================================================
# UTILITIES
# ============================================================================


def _factor_color(factor: str) -> str:
    return FACTOR_COLORS.get(factor, FACTOR_COLORS["unassigned"])


def _abbrev(factor: str) -> str:
    return FACTOR_ABBREV.get(factor, factor[:6].upper())


def _factor_label(factor: str) -> str:
    if factor == UNNAMED_CLUSTER:
        return "Uncategorized Cognition"
    return factor.replace("_", " ").title()


# ============================================================================
# STEP 1: LOAD EXPERT-GUIDED COGNITIVE FACET ASSIGNMENTS
# ============================================================================


def load_theoretical_assignments() -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    Load expert-guided B4 cognitive facet assignments from reference_models_b4.yaml.

    Returns
    -------
    item_to_factor : {item_id: factor_name}
        First-occurrence wins. Used for greedy pairing and MIRT spec.
    item_to_all_factors : {item_id: [factor_name, ...]}
        All factors for each item. Cross-loaders appear under multiple keys.
        Used for cross-loader-aware centroid computation.
    """
    with open(REFERENCE_MODELS_FILE) as fh:
        ref = yaml.safe_load(fh)

    theoretical = ref["theoretical"]
    item_to_factor: dict[str, str] = {}
    item_to_all_factors: dict[str, list[str]] = {}
    cross_loaders: dict[str, list[str]] = {}

    for factor in FACTOR_ORDER:
        for item_id in theoretical.get(factor, []):
            item_to_all_factors.setdefault(item_id, []).append(factor)
            if item_id in item_to_factor:
                cross_loaders.setdefault(item_id, [item_to_factor[item_id]]).append(factor)
            else:
                item_to_factor[item_id] = factor

    if cross_loaders:
        print("  Cross-loading B4 items (first-occurrence cognitive facet wins for pairing/MIRT):")
        for item_id, factor_list in cross_loaders.items():
            all_f = item_to_all_factors[item_id]
            print(f"    {item_id} -> primary: {item_to_factor[item_id]!r}  "
                  f"also contributes centroid to: {[f for f in all_f if f != item_to_factor[item_id]]}")
    else:
        print("  No cross-loading items detected.")

    return item_to_factor, item_to_all_factors


# ============================================================================
# STEPS 2-3: EMBED ITEMS
# ============================================================================


def embed_all_items(
    model,
    item_to_factor: dict[str, str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, set[str]]:
    """
    Parse and embed B4 (theoretical), all 91 B3, and B3 items with dHCP data.

    Returns
    -------
    b4_items, b4_emb, b3_items, b3_emb, b3_data_items, b3_data_id_set
    """
    print("\n  Parsing B4 paper items (restricted to expert-guided cognitive facets)...")
    b4_all = parse_items(str(ITEMS_FILE_B4))
    b4_items = b4_all[b4_all["item_id"].isin(item_to_factor)].copy().reset_index(drop=True)
    b4_emb = embed_items(b4_items, text_column="item_text", model=model)
    print(f"    B4 expert-guided items: {len(b4_items)} "
          f"(of {len(b4_all)} in paper_items_b4.txt)  shape={b4_emb.shape}")

    print("  Parsing all 91 B3 items...")
    b3_items = parse_items(str(ITEMS_FILE_B3_ALL)).reset_index(drop=True)
    b3_emb = embed_items(b3_items, text_column="item_text", model=model)
    print(f"    B3 all items: {len(b3_items)}  shape={b3_emb.shape}")

    print("  Parsing B3 items with dHCP data (34-68)...")
    b3_data_items = parse_items(str(ITEMS_FILE_B3_DATA)).reset_index(drop=True)
    b3_data_id_set: set[str] = set(b3_data_items["item_id"])
    print(f"    B3 items with dHCP data: {len(b3_data_items)}")

    return b4_items, b4_emb, b3_items, b3_emb, b3_data_items, b3_data_id_set


# ============================================================================
# STEPS 4-5: CASCADE PAIRING AND FACTOR ASSIGNMENT
# ============================================================================

# Title-similarity threshold: pairs where sim_to >= this are locked in stage 1.
# 0.75 cleanly separates confident label matches (exact/near-exact title overlap)
# from surface-noise matches; validated on all 39 B4-item pairs.
CASCADE_TITLE_THRESHOLD = 0.75


def _greedy_pairing(sim_matrix: np.ndarray):
    """Greedy exclusive pairing by descending similarity."""
    n_b4, n_b3 = sim_matrix.shape
    flat_order = np.argsort(sim_matrix.ravel())[::-1]
    pairs: list[tuple[int, int, float]] = []
    used_b4: set[int] = set()
    used_b3: set[int] = set()

    for flat_idx in flat_order:
        i, j = divmod(int(flat_idx), n_b3)
        if i in used_b4 or j in used_b3:
            continue
        pairs.append((i, j, float(sim_matrix[i, j])))
        used_b4.add(i)
        used_b3.add(j)
        if len(used_b4) == n_b4 or len(used_b3) == n_b3:
            break

    return pairs, used_b4, used_b3


def _cascade_pairing(
    sim_to: np.ndarray,
    sim_ft: np.ndarray,
    title_threshold: float = CASCADE_TITLE_THRESHOLD,
):
    """
    Two-stage cascade pairing (greedy in both stages).

    Stage 1 — title-locked:
        Pairs where title-only cosine similarity >= title_threshold are locked
        in descending sim_to order.  These are exact / near-exact label matches
        where the title is a reliable proxy (e.g. "Finds Hidden Object" pairs
        trivially with "Finds Hidden Object" at sim=1.0).

    Stage 2 — full-text fallback:
        All remaining B4 items and the still-available B3 pool are paired
        greedily by descending full-text cosine similarity.

    Returns
    -------
    pairs    : list of (b4_idx, b3_idx, sim_title, sim_fulltext)
               sim_title    — title-only cosine similarity (the criterion for
                              stage 1 locking; NaN for fulltext-only pairs is
                              not stored here — both sims are always recorded)
               sim_fulltext — full-text cosine similarity
    stages   : list of 'title_locked' | 'fulltext' (same order as pairs)
    used_b4  : set of B4 row indices that were paired
    used_b3  : set of B3 row indices that were paired
    """
    n_b4, n_b3 = sim_to.shape
    used_b4: set[int] = set()
    used_b3: set[int] = set()
    pairs: list[tuple[int, int, float, float]] = []
    stages: list[str] = []

    # Stage 1: title-confident locks
    flat_order_to = np.argsort(sim_to.ravel())[::-1]
    for flat_idx in flat_order_to:
        i, j = divmod(int(flat_idx), n_b3)
        if sim_to[i, j] < title_threshold:
            break
        if i not in used_b4 and j not in used_b3:
            pairs.append((i, j, float(sim_to[i, j]), float(sim_ft[i, j])))
            stages.append("title_locked")
            used_b4.add(i)
            used_b3.add(j)

    # Stage 2: full-text greedy for remainder
    flat_order_ft = np.argsort(sim_ft.ravel())[::-1]
    for flat_idx in flat_order_ft:
        i, j = divmod(int(flat_idx), n_b3)
        if i not in used_b4 and j not in used_b3:
            pairs.append((i, j, float(sim_to[i, j]), float(sim_ft[i, j])))
            stages.append("fulltext")
            used_b4.add(i)
            used_b3.add(j)
        if len(used_b4) == n_b4:
            break

    return pairs, stages, used_b4, used_b3


def pair_and_assign(
    b4_items: pd.DataFrame,
    b4_emb: np.ndarray,
    b3_items: pd.DataFrame,
    b3_emb: np.ndarray,
    item_to_factor: dict[str, str],
    outdir: Path,
    model=None,
) -> tuple[pd.DataFrame, set[str]]:
    """
    Run cascade pairing and assign paired B3 items to their B4 partner's
    primary (first-wins) theoretical factor.

    Pairing strategy: two-stage greedy cascade.
      Stage 1 (title-locked): pairs where title-only cosine similarity
        >= CASCADE_TITLE_THRESHOLD are locked in first.
      Stage 2 (full-text fallback): remaining items paired by full-text sim.

    Returns
    -------
    paired_df : DataFrame with columns
        b4_item_id, b4_item_title, b3_item_id, b3_item_title,
        sim_title, sim_fulltext, sim_decision, b4_factor, pairing_stage

        sim_title    — title-only cosine similarity (criterion for stage 1)
        sim_fulltext — full-text cosine similarity (criterion for stage 2)
        sim_decision — the similarity that actually drove the pairing decision:
                       sim_title for title-locked pairs, sim_fulltext for
                       fulltext pairs.  Use this column for summary statistics
                       and figures that represent pairing quality.
    unpaired_b3_ids : set[str]
        Item IDs of ALL B3 items that were NOT paired — the residual pool
        used for the LOO unnamed centroid.
    """
    from bayley_nlp.core import embed_items as _embed

    print("  Computing cosine similarity matrices (full text + title)...")
    sim_ft = cosine_similarity(b4_emb, b3_emb)
    b4_emb_to = _embed(b4_items, text_column="item_title", model=model)
    b3_emb_to = _embed(b3_items, text_column="item_title", model=model)
    sim_to = cosine_similarity(b4_emb_to, b3_emb_to)

    pairs, stages, used_b4, used_b3 = _cascade_pairing(sim_to, sim_ft)
    n_locked = stages.count("title_locked")
    n_ft = stages.count("fulltext")
    print(f"  Cascade pairs: {len(pairs)} total  "
          f"({n_locked} title-locked [sim_to>={CASCADE_TITLE_THRESHOLD}], "
          f"{n_ft} full-text fallback)")

    rows = []
    for (i, j, s_title, s_ft), stage in zip(pairs, stages):
        b4_id = b4_items.loc[i, "item_id"]
        sim_decision = s_title if stage == "title_locked" else s_ft
        rows.append(
            {
                "b4_item_id":    b4_id,
                "b4_item_title": b4_items.loc[i, "item_title"],
                "b3_item_id":    b3_items.loc[j, "item_id"],
                "b3_item_title": b3_items.loc[j, "item_title"],
                "sim_title":     s_title,
                "sim_fulltext":  s_ft,
                "sim_decision":  sim_decision,
                "b4_factor":     item_to_factor[b4_id],
                "pairing_stage": stage,
            }
        )

    paired_df = (
        pd.DataFrame(rows)
        .sort_values(["b4_factor", "b4_item_id"])
        .reset_index(drop=True)
    )
    paired_df.to_csv(outdir / "b4_b3_cascade_pairs.csv", index=False)
    print(f"  Mean sim_decision: {paired_df['sim_decision'].mean():.4f}  "
          f"(title-locked: {paired_df.loc[paired_df['pairing_stage']=='title_locked','sim_decision'].mean():.4f}, "
          f"fulltext: {paired_df.loc[paired_df['pairing_stage']=='fulltext','sim_decision'].mean():.4f})"
          if n_locked and n_ft else
          f"  Mean sim_decision: {paired_df['sim_decision'].mean():.4f}")

    # Unpaired B4 items (none expected since n_b4 < n_b3)
    unpaired_b4_idx = sorted(set(range(len(b4_items))) - used_b4)
    if unpaired_b4_idx:
        b4_items.loc[unpaired_b4_idx, ["item_id", "item_title"]].to_csv(
            outdir / "b4_unpaired_items.csv", index=False
        )
        print(f"  WARNING: {len(unpaired_b4_idx)} B4 items were not paired.")

    # ALL unpaired B3 items — residual pool for LOO UC centroid
    unpaired_b3_idx = sorted(set(range(len(b3_items))) - used_b3)
    unpaired_b3_ids: set[str] = set(b3_items.loc[unpaired_b3_idx, "item_id"])
    b3_items.loc[unpaired_b3_idx, ["item_id", "item_title"]].to_csv(
        outdir / "b3_unpaired_items.csv", index=False
    )
    print(f"  Unpaired B3 items (LOO pool): {len(unpaired_b3_idx)}")

    (
        paired_df[["b3_item_id", "b4_factor", "sim_decision"]]
        .rename(columns={"b3_item_id": "item_id", "b4_factor": "assigned_factor",
                         "sim_decision": "pairing_similarity"})
        .to_csv(outdir / "b3_paired_assignments.csv", index=False)
    )

    return paired_df, unpaired_b3_ids


# ============================================================================
# STEP 6a: VISUAL UC CENTROID (for plots only)
# ============================================================================


def compute_visual_unnamed_centroid(
    b3_items: pd.DataFrame,
    b3_emb: np.ndarray,
    unpaired_b3_ids: set[str],
) -> np.ndarray:
    """
    Compute the VISUAL UC centroid as the mean of ALL unpaired B3 items.
    Used solely as a fixed display anchor in plots; actual assignment
    uses per-item LOO centroids (see assign_via_centroids_loo).

    Returns
    -------
    centroid : np.ndarray, shape (d,)
    """
    idxs = [i for i, row in b3_items.iterrows() if row["item_id"] in unpaired_b3_ids]
    if not idxs:
        raise RuntimeError("No unpaired B3 items — cannot form UC centroid.")
    centroid = b3_emb[idxs].mean(axis=0)
    print(f"  Visual UC centroid: mean of {len(idxs)} unpaired B3 items.")
    return centroid


# ============================================================================
# STEP 6b: COGNITIVE FACET CENTROIDS (cross-loader-aware; no UC — per-item)
# ============================================================================


def compute_centroids(
    item_to_factor: dict[str, str],
    item_to_all_factors: dict[str, list[str]],
    b4_items: pd.DataFrame,
    b4_emb: np.ndarray,
    paired_df: pd.DataFrame,
    b3_items: pd.DataFrame,
    b3_emb: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Compute cognitive facet centroids for three methods.

    Cross-loader awareness:
        B4 items listed in multiple expert-guided facets contribute their
        embedding to ALL those facets' centroids (not just the first-listed).
        The MIRT spec still uses first-occurrence assignment for pairing.

    UC is NOT included here; it is computed per-item during assignment
    (see assign_via_centroids_loo).

    Returns
    -------
    {method_name: {factor_name: centroid_vector}}
    """
    b4_id_to_idx: dict[str, int] = {row["item_id"]: i for i, row in b4_items.iterrows()}
    b3_id_to_idx: dict[str, int] = {row["item_id"]: i for i, row in b3_items.iterrows()}

    b4_centroids: dict[str, np.ndarray] = {}
    b3_centroids: dict[str, np.ndarray] = {}
    mixed_centroids: dict[str, np.ndarray] = {}

    for factor in FACTOR_ORDER:
        # B4 vectors: cross-loader-aware — include item if factor is in ANY of its
        # listed factors, not just the first one.
        b4_idxs = [
            b4_id_to_idx[iid]
            for iid, factors in item_to_all_factors.items()
            if factor in factors and iid in b4_id_to_idx
        ]
        b4_vecs = b4_emb[b4_idxs] if b4_idxs else np.empty((0, b4_emb.shape[1]))

        # B3 vectors: paired items are locked to first-wins factor (pairing is fixed).
        paired_b3_ids = paired_df.loc[paired_df["b4_factor"] == factor, "b3_item_id"].tolist()
        b3_idxs = [b3_id_to_idx[iid] for iid in paired_b3_ids if iid in b3_id_to_idx]
        b3_vecs = b3_emb[b3_idxs] if b3_idxs else np.empty((0, b3_emb.shape[1]))

        if len(b4_vecs):
            b4_centroids[factor] = b4_vecs.mean(axis=0)
        if len(b3_vecs):
            b3_centroids[factor] = b3_vecs.mean(axis=0)

        parts = [v for v in (b4_vecs, b3_vecs) if len(v)]
        if parts:
            mixed_centroids[factor] = np.vstack(parts).mean(axis=0)

    print(f"    b4_centroids    : {len(b4_centroids)} cognitive facets")
    print(f"    b3_centroids    : {len(b3_centroids)} cognitive facets")
    print(f"    mixed_centroids : {len(mixed_centroids)} cognitive facets")

    # Report cross-loader impact
    cross_load_items = {iid: fs for iid, fs in item_to_all_factors.items() if len(fs) > 1}
    if cross_load_items:
        print(f"    Cross-loaders contributing to multiple centroids: {len(cross_load_items)}")
        for iid, fs in cross_load_items.items():
            print(f"      {iid}: {' + '.join(_abbrev(f) for f in fs)}")

    return {
        "b4_centroids": b4_centroids,
        "b3_centroids": b3_centroids,
        "mixed_centroids": mixed_centroids,
    }


# ============================================================================
# STEP 6c: LOO CENTROID ASSIGNMENT (runs over ALL unpaired B3)
# ============================================================================


def assign_via_centroids_loo(
    unpaired_items: pd.DataFrame,
    unpaired_emb: np.ndarray,
    ef_centroids: dict[str, np.ndarray],
    b3_emb: np.ndarray,
    b3_id_to_idx: dict[str, int],
    unpaired_b3_ids: set[str],
) -> pd.DataFrame:
    """
    Assign unpaired B3 items using a Leave-One-Out UC centroid.

    v5 change: caller passes ALL unpaired B3 items (not just dHCP 34-68).
    The LOO pool is still all unpaired B3 items; dHCP filtering happens
    in main() after this call.

    For each item being evaluated:
      - Compute LOO UC centroid = mean of all unpaired B3 embeddings
        EXCLUDING this item's own embedding.
      - Compete against EF factor centroids + this LOO centroid.
      - Assign to the nearest centroid.

    LOO centroid computed efficiently via precomputed sum:
      loo_centroid = (pool_sum - item_vec) / (N - 1)

    Parameters
    ----------
    unpaired_items : DataFrame of ALL unpaired B3 items.
    unpaired_emb   : Their embedding matrix (same row order).
    ef_centroids   : EF factor centroids dict (UC not included).
    b3_emb         : Full B3 embedding matrix.
    b3_id_to_idx   : item_id -> row index in b3_emb.
    unpaired_b3_ids: All B3 item IDs not paired (the LOO pool).

    Returns
    -------
    DataFrame with columns:
      item_id, item_title, assigned_factor, assignment_method,
      centroid_similarity, sim_ATT, sim_WM, sim_GDPS, sim_FS, sim_HOP, sim_UC,
      best_ef_factor, best_ef_sim, second_ef_factor, second_ef_sim, uc_margin

    best_ef_factor / best_ef_sim : highest-similarity EF cognitive facet (UC excluded).
    second_ef_factor / second_ef_sim : 2nd-highest EF cognitive facet.
    uc_margin : sim_UC - best_ef_sim  (positive => UC was genuinely nearest).
    """
    factors_available = [f for f in FACTOR_ORDER if f in ef_centroids]
    ef_centroid_matrix = np.vstack([ef_centroids[f] for f in factors_available])

    # Precompute sum and count of all unpaired B3 embeddings (LOO pool)
    pool_idxs = [b3_id_to_idx[iid] for iid in unpaired_b3_ids if iid in b3_id_to_idx]
    pool_emb = b3_emb[pool_idxs]       # shape (N_pool, d)
    pool_sum = pool_emb.sum(axis=0)    # shape (d,)
    n_pool = len(pool_idxs)

    rows = []
    for i, (_, item_row) in enumerate(unpaired_items.iterrows()):
        item_id = item_row["item_id"]
        item_vec = unpaired_emb[i]     # shape (d,)

        # --- LOO UC centroid ---
        if item_id in unpaired_b3_ids and item_id in b3_id_to_idx:
            loo_sum = pool_sum - item_vec
            loo_n = n_pool - 1
        else:
            # Item not in pool — safe fallback to full mean
            loo_sum = pool_sum
            loo_n = n_pool

        loo_centroid = loo_sum / max(loo_n, 1)  # shape (d,)

        # --- Similarities ---
        item_2d = item_vec.reshape(1, -1)
        ef_sims = cosine_similarity(item_2d, ef_centroid_matrix)[0]  # shape (n_ef,)
        uc_sim = float(cosine_similarity(item_2d, loo_centroid.reshape(1, -1))[0, 0])

        all_sims = list(ef_sims) + [uc_sim]
        all_factors = factors_available + [UNNAMED_CLUSTER]
        best_idx = int(np.argmax(all_sims))

        # --- EF rank 1 and rank 2 (UC excluded) ---
        ef_rank = np.argsort(ef_sims)[::-1]  # descending rank among EF facets only
        best_ef_factor = factors_available[ef_rank[0]]
        best_ef_sim = float(ef_sims[ef_rank[0]])
        if len(ef_rank) >= 2:
            second_ef_factor = factors_available[ef_rank[1]]
            second_ef_sim = float(ef_sims[ef_rank[1]])
        else:
            second_ef_factor = None
            second_ef_sim = float("nan")
        uc_margin = uc_sim - best_ef_sim

        row: dict = {
            "item_id": item_id,
            "item_title": item_row["item_title"],
            "assigned_factor": all_factors[best_idx],
            "assignment_method": "centroid_loo",
            "centroid_similarity": float(all_sims[best_idx]),
            "best_ef_factor": best_ef_factor,
            "best_ef_sim": best_ef_sim,
            "second_ef_factor": second_ef_factor,
            "second_ef_sim": second_ef_sim,
            "uc_margin": uc_margin,
        }
        for j, f in enumerate(factors_available):
            row[f"sim_{_abbrev(f)}"] = float(ef_sims[j])
        row[f"sim_{_abbrev(UNNAMED_CLUSTER)}"] = uc_sim   # sim_UC
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# BUILD FULL 34-68 ASSIGNMENT TABLE
# ============================================================================


def build_full_34_68_assignment(
    paired_df: pd.DataFrame,
    centroid_assignments: pd.DataFrame,
    b3_data_items: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge paired assignments (34-68 subset) and LOO centroid assignments
    into a single table, preserving the original item ordering.
    Items assigned to UC keep that label here.
    """
    data_ids = set(b3_data_items["item_id"])

    paired_in_data = (
        paired_df.loc[paired_df["b3_item_id"].isin(data_ids), ["b3_item_id", "b4_factor", "sim_decision"]]
        .rename(columns={"b3_item_id": "item_id", "b4_factor": "assigned_factor",
                         "sim_decision": "assignment_score"})
        .assign(assignment_method="cascade_pairing")
    )

    centroid_in_data = (
        centroid_assignments[["item_id", "assigned_factor", "centroid_similarity"]]
        .rename(columns={"centroid_similarity": "assignment_score"})
        .assign(assignment_method="centroid_loo")
    )

    combined = pd.concat([paired_in_data, centroid_in_data], ignore_index=True)
    combined = (
        combined
        .set_index("item_id")
        .reindex(b3_data_items["item_id"])
        .reset_index()
    )
    return combined[["item_id", "assigned_factor", "assignment_method", "assignment_score"]]


# ============================================================================
# STEP 7: SAVE MODEL SPECIFICATIONS
# ============================================================================


def save_model_spec(
    full_assignment: pd.DataFrame,
    centroid_assignments_34_68: pd.DataFrame,
    b3_data_items: pd.DataFrame,
    centroid_method: str,
    paired_df: pd.DataFrame,
    outdir: Path,
) -> tuple[str, list[str], list[str], dict[str, str]]:
    """
    Save the B3 model specification.

    v6 changes vs v5:
      - LOO-UC items are REDIRECTED to their best_ef_factor (not placed in UC).
      - Cognitive facets with < MIN_ITEMS_PER_EF_FACTOR items in 34-68 dHCP set
        are EXCLUDED entirely (their items are dropped from the model).
      - No UC factor in the model spec.
      - Factor numbering: valid EF facets get F1..Fn only.

    Returns
    -------
    model_name          : str  — key inserted into model_specs.yaml.
    sparse_ef_factors   : list — cognitive facets that were excluded (too few items).
    excluded_items      : list — item IDs excluded because their facet was sparse.
    redirected_items    : dict — {item_id: best_ef_factor} for LOO-UC redirected items.
    """
    model_slug = MODEL.replace("/", "_").replace("-", "_")
    model_name = f"{model_slug}_b3_expert_guided_v6_cog_{centroid_method}"

    b3_data_ids = b3_data_items["item_id"].tolist()
    id_to_pos = {iid: i + 1 for i, iid in enumerate(b3_data_ids)}  # 1-based

    # Build lookup for LOO results (has best_ef_factor column)
    ca_lookup = centroid_assignments_34_68.set_index("item_id")

    # ------------------------------------------------------------------
    # Phase 1: classify items — redirect LOO-UC to best_ef_factor
    # ------------------------------------------------------------------
    ef_items: dict[str, list[str]] = {f: [] for f in FACTOR_ORDER}
    redirected_items: dict[str, str] = {}  # item_id -> best_ef_factor

    for _, row in full_assignment.dropna(subset=["assigned_factor"]).iterrows():
        item_id = row["item_id"]
        factor = row["assigned_factor"]
        if factor == UNNAMED_CLUSTER:
            # Redirect to best EF cognitive facet
            if item_id in ca_lookup.index and "best_ef_factor" in ca_lookup.columns:
                best_ef = ca_lookup.loc[item_id, "best_ef_factor"]
                if pd.notna(best_ef) and best_ef in ef_items:
                    ef_items[best_ef].append(item_id)
                    redirected_items[item_id] = best_ef
                else:
                    print(f"    WARNING: UC item {item_id!r} has no valid best_ef_factor "
                          f"({best_ef!r}) — dropped from model spec!")
            else:
                print(f"    WARNING: UC item {item_id!r} not in LOO lookup — "
                      f"dropped from model spec!")
        elif factor in ef_items and item_id in id_to_pos:
            ef_items[factor].append(item_id)

    # ------------------------------------------------------------------
    # Phase 2: identify sparse cognitive facets — exclude their items
    # ------------------------------------------------------------------
    sparse_ef_factors: list[str] = []
    excluded_items: list[str] = []

    for factor in FACTOR_ORDER:
        if len(ef_items[factor]) < MIN_ITEMS_PER_EF_FACTOR:
            sparse_ef_factors.append(factor)
            excluded_items.extend(ef_items.pop(factor))
        else:
            ef_items[factor] = sorted(ef_items[factor], key=lambda iid: id_to_pos.get(iid, 9999))

    if sparse_ef_factors:
        print(f"    Sparse cognitive facets (< {MIN_ITEMS_PER_EF_FACTOR} items) → excluded:")
        for f in sparse_ef_factors:
            print(f"      {_abbrev(f)} ({_factor_label(f)}): "
                  f"{len([i for i in excluded_items if i in ef_items.get(f, [])])} items excluded")

    # ------------------------------------------------------------------
    # Phase 3: build final factor list with dynamic numbering (no UC)
    # ------------------------------------------------------------------
    valid_ef_factors = [f for f in FACTOR_ORDER if f not in sparse_ef_factors]
    factor_key_map = {factor: f"F{i + 1}" for i, factor in enumerate(valid_ef_factors)}
    n_factors = len(valid_ef_factors)

    factor_items_yaml: dict[str, list] = {f"F{i + 1}": [] for i in range(n_factors)}
    factor_indices_yaml: dict[str, list] = {f"F{i + 1}": [] for i in range(n_factors)}
    factor_names_yaml: dict[str, str] = {
        f"F{i + 1}": factor for i, factor in enumerate(valid_ef_factors)
    }

    for factor in valid_ef_factors:
        fkey = factor_key_map[factor]
        for iid in ef_items[factor]:
            factor_items_yaml[fkey].append(iid)
            factor_indices_yaml[fkey].append(id_to_pos[iid])

    # ------------------------------------------------------------------
    # Phase 4: assemble metadata
    # ------------------------------------------------------------------
    data_ids_set = set(b3_data_ids)
    n_paired = int(paired_df["b3_item_id"].isin(data_ids_set).sum())
    # n_facet_direct: LOO-assigned directly to an EF facet (did not pass through UC)
    n_facet_direct = int(
        (centroid_assignments_34_68["assigned_factor"] != UNNAMED_CLUSTER).sum()
    )
    # n_uc_redirected: LOO-assigned to UC, then redirected to best EF facet
    n_uc_redirected = len(redirected_items)
    # n_centroid_named: all LOO-derived items that end up in the model
    n_centroid_named = n_facet_direct + n_uc_redirected
    n_excluded_sparse = len(excluded_items)
    n_total = sum(len(v) for v in factor_items_yaml.values())

    spec: dict = {
        "source": "b3_expert_guided_mapping_v6_cog",
        "model": MODEL,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "reference_model": "Aylward et al. (2022) expert-guided cognitive facets",
        "n_factors": n_factors,
        "n_items": n_total,
        "centroid_method": centroid_method,
        "uc_policy": "LOO-UC items redirected to best cognitive facet; sparse facets excluded",
        "min_items_per_ef_factor": MIN_ITEMS_PER_EF_FACTOR,
        "n_paired_items_in_data": n_paired,
        "n_centroid_assigned_items": n_centroid_named,
        "n_facet_direct": n_facet_direct,
        "n_uc_redirected": n_uc_redirected,
        "uc_redirected_items": redirected_items,
        "n_excluded_sparse_ef": n_excluded_sparse,
        "excluded_sparse_ef_items": excluded_items,
        "sparse_ef_factors": sparse_ef_factors,
        "final_ef_factors": valid_ef_factors,
        "mean_pairing_similarity": float(round(paired_df["sim_decision"].mean(), 4)),
        "factor_names": factor_names_yaml,
        "factor_items": factor_items_yaml,
        "factor_indices": factor_indices_yaml,
        "factor_sizes": {fkey: len(v) for fkey, v in factor_items_yaml.items()},
        "description": (
            f"B3 items 34-68 assigned to expert-guided cognitive facets of Aylward et al. (2022). "
            f"{n_paired} items via cascade B4-B3 pairing (title-locked + full-text fallback); "
            f"{n_facet_direct} unpaired items via LOO ({centroid_method}) directly to EF facets; "
            f"{n_uc_redirected} LOO-UC items redirected to best cognitive facet "
            f"(total LOO-in-model: {n_centroid_named}); "
            f"{n_excluded_sparse} items excluded from sparse cognitive facets "
            f"({', '.join(_abbrev(f) for f in sparse_ef_factors) or 'none'}). "
            f"Cross-loaders contributed embeddings to all listed facets' centroids."
        ),
    }

    with open(MODEL_SPECS_FILE) as fh:
        all_specs = yaml.safe_load(fh) or {}
    all_specs[model_name] = spec
    with open(MODEL_SPECS_FILE, "w") as fh:
        yaml.dump(all_specs, fh, default_flow_style=False, sort_keys=False)

    method_dir = outdir / centroid_method
    with open(method_dir / "b3_model_spec.yaml", "w") as fh:
        yaml.dump({model_name: spec}, fh, default_flow_style=False, sort_keys=False)

    full_assignment.to_csv(method_dir / "full_34_68_assignments.csv", index=False)

    print(f"    Saved spec: {model_name}")
    print(f"      Final cognitive facets: {[factor_key_map[f]+':'+_abbrev(f) for f in valid_ef_factors]}")
    print(f"      Factor sizes  : {spec['factor_sizes']}")

    return model_name, sparse_ef_factors, excluded_items, redirected_items


# ============================================================================
# STEP 7b: UC SIMILARITY PROFILE (diagnostics — items not in final model)
# ============================================================================


def compute_uc_profile(
    centroid_assignments_34_68: pd.DataFrame,
    b3_data_items: pd.DataFrame,
    sparse_ef_factors: list[str],
    excluded_items: list[str],
    redirected_items: dict[str, str],
    outdir: Path,
    centroid_method: str,
) -> pd.DataFrame:
    """
    Build a per-item similarity profile for all items that were candidates
    for the UC (uncategorized cognition) assignment.

    v6 sources (diagnostic, not in model):
      loo_uc_redirected : LOO assigned UC; redirected to best_ef_factor in model spec.
      sparse_ef_excluded: In a sparse cognitive facet; excluded from model entirely.

    Columns
    -------
    item_id, item_title, source, final_factor,
    uc_sim, best_ef_factor, best_ef_sim, margin
      margin = uc_sim - best_ef_sim
    """
    uc_sim_col = f"sim_{_abbrev(UNNAMED_CLUSTER)}"   # sim_UC
    sim_ef_cols = {f: f"sim_{_abbrev(f)}" for f in FACTOR_ORDER}

    uc_redirected_set = set(redirected_items.keys())
    excluded_set = set(excluded_items)

    ca_lookup = centroid_assignments_34_68.set_index("item_id")
    title_lookup = dict(zip(b3_data_items["item_id"], b3_data_items["item_title"]))

    rows = []

    # --- LOO-UC redirected items ---
    for iid in uc_redirected_set:
        if iid not in ca_lookup.index:
            continue
        row = ca_lookup.loc[iid]
        uc_sim = float(row.get(uc_sim_col, float("nan")))
        ef_sims = {}
        for f in FACTOR_ORDER:
            col = sim_ef_cols[f]
            if col in row.index:
                ef_sims[f] = float(row[col])
        if ef_sims:
            best_ef_factor = max(ef_sims, key=ef_sims.__getitem__)
            best_ef_sim = ef_sims[best_ef_factor]
        else:
            best_ef_factor = None
            best_ef_sim = float("nan")
        margin = (uc_sim - best_ef_sim
                  if not (np.isnan(uc_sim) or np.isnan(best_ef_sim))
                  else float("nan"))
        rows.append({
            "item_id": iid,
            "item_title": row.get("item_title", title_lookup.get(iid, "")),
            "source": "loo_uc_redirected",
            "final_factor": redirected_items.get(iid, best_ef_factor),
            "uc_sim": uc_sim,
            "best_ef_factor": best_ef_factor,
            "best_ef_sim": best_ef_sim,
            "margin": margin,
        })

    # --- Sparse-EF excluded items ---
    for iid in excluded_set:
        # These were paired items — no LOO centroid sim scores available
        rows.append({
            "item_id": iid,
            "item_title": title_lookup.get(iid, ""),
            "source": "sparse_ef_excluded",
            "final_factor": None,
            "uc_sim": float("nan"),
            "best_ef_factor": None,
            "best_ef_sim": float("nan"),
            "margin": float("nan"),
        })

    profile_df = (
        pd.DataFrame(rows)
        .sort_values(["source", "margin"], ascending=[True, False])
        .reset_index(drop=True)
    )
    profile_df.to_csv(outdir / centroid_method / "uc_similarity_profile.csv", index=False)
    return profile_df


# ============================================================================
# STEP 8: VISUALIZATIONS
# ============================================================================


def plot_pairing_mds(
    b4_items: pd.DataFrame,
    b4_emb: np.ndarray,
    b3_items: pd.DataFrame,
    b3_emb: np.ndarray,
    paired_df: pd.DataFrame,
    outdir: Path,
) -> None:
    """MDS projection of paired B4+B3 items, colored by theoretical factor."""
    print("  [1/6] MDS pairing plot...")

    b4_id_to_idx = {row["item_id"]: i for i, row in b4_items.iterrows()}
    b3_id_to_idx = {row["item_id"]: i for i, row in b3_items.iterrows()}

    b4_idx = [b4_id_to_idx[iid] for iid in paired_df["b4_item_id"]]
    b3_idx = [b3_id_to_idx[iid] for iid in paired_df["b3_item_id"]]

    combined_emb = np.vstack([b4_emb[b4_idx], b3_emb[b3_idx]])
    dist = np.clip(1 - cosine_similarity(combined_emb), 0, None)

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=4)
    coords = mds.fit_transform(dist)

    n_pairs = len(paired_df)
    b4_coords = coords[:n_pairs]
    b3_coords = coords[n_pairs:]

    fig, ax = plt.subplots(figsize=(16, 11))

    for i, row in paired_df.iterrows():
        color = _factor_color(row["b4_factor"])
        ax.plot(
            [b4_coords[i, 0], b3_coords[i, 0]],
            [b4_coords[i, 1], b3_coords[i, 1]],
            color=color, alpha=0.30, linewidth=0.9, zorder=1,
        )

    for i, row in paired_df.iterrows():
        color = _factor_color(row["b4_factor"])
        ax.scatter(*b4_coords[i], c=color, s=110, marker="^", zorder=4,
                   edgecolors="white", linewidths=0.7)
        ax.scatter(*b3_coords[i], c=color, s=80, marker="o", zorder=4,
                   edgecolors="white", linewidths=0.7)
        ax.annotate(row["b4_item_id"], b4_coords[i], fontsize=4.5,
                    ha="center", va="bottom", xytext=(0, 5),
                    textcoords="offset points", color="#333333")
        ax.annotate(row["b3_item_id"], b3_coords[i], fontsize=4.5,
                    ha="center", va="top", xytext=(0, -5),
                    textcoords="offset points", color="#333333")

    factor_handles = [
        mpatches.Patch(color=_factor_color(f), label=f"{_abbrev(f)} — {_factor_label(f)}")
        for f in FACTOR_ORDER
    ]
    type_handles = [
        plt.Line2D([0], [0], marker="^", color="gray", markersize=9,
                   linestyle="None", label="B4 item (expert-guided)"),
        plt.Line2D([0], [0], marker="o", color="gray", markersize=8,
                   linestyle="None", label="B3 item (paired)"),
    ]
    ax.legend(handles=factor_handles + type_handles,
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, framealpha=0.9)
    ax.set_title(
        "MDS Projection: B4↔B3 Greedy Pairs by Expert-Guided Cognitive Facet\n"
        "(triangles = B4 items; circles = paired B3 items; color = cognitive facet)",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("MDS-1", fontsize=9)
    ax.set_ylabel("MDS-2", fontsize=9)
    plt.tight_layout()
    out = outdir / "figures" / "01_pairing_mds.png"
    fig.savefig(str(out), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved -> {out.name}")


def _v6_factor(item_id: str, paired_map: dict, ca_all_lookup) -> str:
    """Return the v6 final factor for an item: paired factor, LOO EF, or UC-redirected."""
    if item_id in paired_map:
        return paired_map[item_id]
    if ca_all_lookup is not None and item_id in ca_all_lookup.index:
        row = ca_all_lookup.loc[item_id]
        af = row.get("assigned_factor", None)
        if af == UNNAMED_CLUSTER:
            bef = row.get("best_ef_factor", None)
            return bef if pd.notna(bef) else UNNAMED_CLUSTER
        return af if pd.notna(af) else UNNAMED_CLUSTER
    return UNNAMED_CLUSTER


def plot_b3_factor_umap(
    b3_items: pd.DataFrame,
    b3_emb: np.ndarray,
    paired_df: pd.DataFrame,
    b3_data_id_set: set[str],
    outdir: Path,
    ca_all_df: pd.DataFrame | None = None,
) -> None:
    """UMAP of all 91 B3 items colored by v6 final cognitive facet assignment."""
    print("  [2/6] B3 factor assignment UMAP...")

    paired_map = dict(zip(paired_df["b3_item_id"], paired_df["b4_factor"]))
    ca_all_lookup = ca_all_df.set_index("item_id") if ca_all_df is not None else None

    reducer = get_reducer("umap", verbose=False, n_components=2,
                          random_state=42, n_neighbors=10, min_dist=0.05,
                          metric="cosine")
    coords = reducer.fit_transform(b3_emb)

    fig, ax = plt.subplots(figsize=(16, 11))
    for i, row in b3_items.iterrows():
        item_id = row["item_id"]
        factor = _v6_factor(item_id, paired_map, ca_all_lookup)
        color = _factor_color(factor)
        has_data = item_id in b3_data_id_set
        marker = "v" if item_id in paired_map else ("D" if has_data else "o")
        size = 100 if has_data else 50
        alpha = 0.92 if has_data else 0.55
        ax.scatter(coords[i, 0], coords[i, 1], c=color, s=size, marker=marker,
                   alpha=alpha, edgecolors="white", linewidths=0.5, zorder=3)
        ax.annotate(item_id, (coords[i, 0], coords[i, 1]), fontsize=4.5,
                    ha="center", va="bottom", xytext=(0, 4),
                    textcoords="offset points", color="#333333", zorder=4)

    uc_abbrev = _abbrev(UNNAMED_CLUSTER)
    uc_label = _factor_label(UNNAMED_CLUSTER)
    factor_handles = [
        mpatches.Patch(color=_factor_color(f), label=_abbrev(f))
        for f in FACTOR_ORDER
    ] + [mpatches.Patch(color=FACTOR_COLORS[UNNAMED_CLUSTER], label=f"{uc_abbrev} (no assignment)")]
    type_handles = [
        plt.Line2D([0], [0], marker="v", color="gray", markersize=8,
                   linestyle="None", label="Paired (expert-guided)"),
        plt.Line2D([0], [0], marker="D", color="gray", markersize=7,
                   linestyle="None", label="LOO-assigned dHCP"),
        plt.Line2D([0], [0], marker="o", color="gray", markersize=6,
                   linestyle="None", label="Non-dHCP"),
    ]
    ax.legend(handles=factor_handles + type_handles,
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, framealpha=0.9)
    ax.set_title(
        "UMAP: All 91 B3 Items — v6_cog Final Cognitive Facet Assignment\n"
        "(▼ paired; ◆ LOO dHCP; ○ non-dHCP; color = final model facet incl. UC-redirected)",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("UMAP-1", fontsize=9)
    ax.set_ylabel("UMAP-2", fontsize=9)
    plt.tight_layout()
    out = outdir / "figures" / "02_b3_factor_assignment_umap.png"
    fig.savefig(str(out), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved -> {out.name}")


def plot_b3_factor_dr_variants(
    b3_items: pd.DataFrame,
    b3_emb: np.ndarray,
    paired_df: pd.DataFrame,
    b3_data_id_set: set[str],
    outdir: Path,
    ca_all_df: pd.DataFrame | None = None,
) -> None:
    """2×2 grid: B3 factor coloring across four DR methods (v6 final assignments)."""
    print("  [new] B3 factor assignment DR variants (2×2 grid)...")

    paired_map = dict(zip(paired_df["b3_item_id"], paired_df["b4_factor"]))
    ca_all_lookup = ca_all_df.set_index("item_id") if ca_all_df is not None else None

    DR_CONFIGS = [
        ("UMAP",   "umap",
         dict(n_components=2, random_state=42, n_neighbors=15,
              min_dist=0.08, metric="cosine", verbose=False)),
        ("t-SNE",  "tsne",
         dict(n_components=2, random_state=42, perplexity=25, verbose=0)),
        ("PCA",    "pca",    dict(n_components=2)),
        ("Isomap", "isomap", dict(n_components=2, n_neighbors=10)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    axes_flat = axes.flatten()

    for ax, (method_name, method_key, kwargs) in zip(axes_flat, DR_CONFIGS):
        try:
            reducer = get_reducer(method_key, **kwargs)
            coords = reducer.fit_transform(b3_emb)
        except Exception as exc:
            ax.text(0.5, 0.5, f"Error:\n{exc}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="red")
            ax.set_title(method_name, fontsize=11, fontweight="bold")
            continue

        for i, row in b3_items.iterrows():
            item_id = row["item_id"]
            factor = _v6_factor(item_id, paired_map, ca_all_lookup)
            color = _factor_color(factor)
            has_data = item_id in b3_data_id_set
            marker = "v" if item_id in paired_map else ("D" if has_data else "o")
            size = 110 if has_data else 55
            alpha = 0.92 if has_data else 0.55
            ax.scatter(coords[i, 0], coords[i, 1], c=color, s=size, marker=marker,
                       alpha=alpha, edgecolors="white", linewidths=0.5, zorder=3)
            ax.annotate(item_id, (coords[i, 0], coords[i, 1]), fontsize=4.5,
                        ha="center", va="bottom", xytext=(0, 4),
                        textcoords="offset points", color="#333333", zorder=4)

        ax.set_title(method_name, fontsize=12, fontweight="bold")
        ax.set_xlabel(f"{method_name}-1", fontsize=9)
        ax.set_ylabel(f"{method_name}-2", fontsize=9)
        ax.tick_params(labelsize=7)

    uc_abbrev = _abbrev(UNNAMED_CLUSTER)
    factor_handles = [
        mpatches.Patch(color=_factor_color(f), label=_abbrev(f))
        for f in FACTOR_ORDER
    ] + [mpatches.Patch(color=FACTOR_COLORS[UNNAMED_CLUSTER], label=f"{uc_abbrev} (no assignment)")]
    type_handles = [
        plt.Line2D([0], [0], marker="v", color="gray", markersize=8,
                   linestyle="None", label="Paired"),
        plt.Line2D([0], [0], marker="D", color="gray", markersize=7,
                   linestyle="None", label="LOO dHCP"),
        plt.Line2D([0], [0], marker="o", color="gray", markersize=6,
                   linestyle="None", label="Non-dHCP"),
    ]
    fig.legend(handles=factor_handles + type_handles,
               loc="lower center", bbox_to_anchor=(0.5, -0.04),
               fontsize=9, framealpha=0.9, ncol=5)
    fig.suptitle(
        "B3 Items: v6_cog Final Cognitive Facet Assignment Across DR Methods\n"
        "(▼ paired; ◆ LOO dHCP; ○ non-dHCP; color = final model facet incl. UC-redirected)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out = outdir / "figures" / "02b_b3_factor_dr_variants.png"
    fig.savefig(str(out), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved -> {out.name}")


def plot_pairing_three_subplots(
    b4_items: pd.DataFrame,
    b4_emb: np.ndarray,
    b3_items: pd.DataFrame,
    b3_emb: np.ndarray,
    paired_df: pd.DataFrame,
    b3_data_id_set: set[str],
    all_ef_centroids: dict[str, dict[str, np.ndarray]],
    all_centroid_assignments: dict[str, pd.DataFrame],
    visual_unnamed_centroid: np.ndarray,
    outdir: Path,
) -> None:
    """
    Three-panel MDS (one panel per centroid method: b4, b3, mixed).

    Each panel shows:
      △  B4 paired items (up-triangle)
      ▽  B3 paired items (down-triangle)
      —  very faint solid lines connecting each B4↔B3 pair
      ★  EF factor centroids (method-specific per panel)
      ◆  Single visual UC centroid (grey; same position across panels)
      ○  unpaired dHCP items (34-68) colored by their LOO assignment
      --  dashed line from each unpaired item to assigned centroid
         (grey dashed for UC items → visual UC centroid)

    The MDS projection is shared across all three panels.
    The visual UC centroid is included once in the joint embedding.
    """
    print("  [new] Three-subplot pairing + centroid assignment overview...")

    b4_id_to_idx = {row["item_id"]: i for i, row in b4_items.iterrows()}
    b3_id_to_idx = {row["item_id"]: i for i, row in b3_items.iterrows()}
    b4_to_factor = dict(zip(paired_df["b4_item_id"], paired_df["b4_factor"]))
    b4_paired_order = paired_df["b4_item_id"].tolist()
    b4_coord_idx = {iid: i for i, iid in enumerate(b4_paired_order)}

    factors_available = [
        f for f in FACTOR_ORDER
        if all(f in all_ef_centroids[m] for m in CENTROID_METHODS)
    ]
    n_factors = len(factors_available)

    # Joint MDS:
    # [B4 paired | B3 all | per-method EF centroids | visual_UC_centroid (once)]
    b4_paired_emb = np.vstack([b4_emb[b4_id_to_idx[iid]] for iid in b4_paired_order])
    ef_centroid_blocks = [
        np.vstack([all_ef_centroids[m][f] for f in factors_available])
        for m in CENTROID_METHODS
    ]
    combined = np.vstack(
        [b4_paired_emb, b3_emb]
        + ef_centroid_blocks
        + [visual_unnamed_centroid.reshape(1, -1)]
    )
    dist = np.clip(1 - cosine_similarity(combined), 0, None)

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=4)
    all_mds_coords = mds.fit_transform(dist)

    n_b4 = len(b4_paired_order)
    n_b3 = len(b3_items)
    b4_paired_coords = all_mds_coords[:n_b4]
    b3_all_coords = all_mds_coords[n_b4:n_b4 + n_b3]

    ef_centroid_coords_by_method: dict[str, dict[str, np.ndarray]] = {}
    for mi, method in enumerate(CENTROID_METHODS):
        offset = n_b4 + n_b3 + mi * n_factors
        ef_centroid_coords_by_method[method] = {
            f: all_mds_coords[offset + fi] for fi, f in enumerate(factors_available)
        }

    visual_uc_coord = all_mds_coords[-1]
    uc_abbrev = _abbrev(UNNAMED_CLUSTER)

    METHOD_TITLES = {
        "b4_centroids":    "B4 Centroids\n(mean of expert-guided B4 item embeddings)",
        "b3_centroids":    "B3 Centroids\n(mean of paired B3 item embeddings)",
        "mixed_centroids": "Mixed Centroids\n(B4 expert-guided + paired B3 pooled)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    for ax, method in zip(axes, CENTROID_METHODS):
        ef_coord_map = ef_centroid_coords_by_method[method]
        ca_df = all_centroid_assignments[method]

        # 1. Very faint B4↔B3 pair lines
        for _, row in paired_df.iterrows():
            color = _factor_color(row["b4_factor"])
            ci_b3 = b3_id_to_idx[row["b3_item_id"]]
            ci_b4 = b4_coord_idx[row["b4_item_id"]]
            ax.plot(
                [b4_paired_coords[ci_b4, 0], b3_all_coords[ci_b3, 0]],
                [b4_paired_coords[ci_b4, 1], b3_all_coords[ci_b3, 1]],
                color=color, alpha=0.10, linewidth=0.6, zorder=1,
            )

        # 2. B3 paired items (▽)
        for _, row in paired_df.iterrows():
            ci = b3_id_to_idx[row["b3_item_id"]]
            color = _factor_color(row["b4_factor"])
            ax.scatter(b3_all_coords[ci, 0], b3_all_coords[ci, 1],
                       c=color, s=105, marker="v", zorder=4,
                       edgecolors="white", linewidths=0.6)
            ax.annotate(row["b3_item_id"],
                        (b3_all_coords[ci, 0], b3_all_coords[ci, 1]),
                        fontsize=4.0, ha="center", va="top", xytext=(0, -4),
                        textcoords="offset points", color="#444444")

        # 3. B4 paired items (△)
        for pi, b4_id in enumerate(b4_paired_order):
            color = _factor_color(b4_to_factor[b4_id])
            ax.scatter(b4_paired_coords[pi, 0], b4_paired_coords[pi, 1],
                       c=color, s=105, marker="^", zorder=4,
                       edgecolors="white", linewidths=0.6)
            ax.annotate(b4_id,
                        (b4_paired_coords[pi, 0], b4_paired_coords[pi, 1]),
                        fontsize=4.0, ha="center", va="bottom", xytext=(0, 4),
                        textcoords="offset points", color="#444444")

        # 4. Dashed lines: unpaired dHCP → assigned centroid
        for _, row in ca_df.iterrows():
            iid = row["item_id"]
            factor = row["assigned_factor"]
            color = _factor_color(factor)
            ci = b3_id_to_idx.get(iid)
            if ci is None:
                continue
            target_coord = (
                visual_uc_coord if factor == UNNAMED_CLUSTER
                else ef_coord_map.get(factor)
            )
            if target_coord is not None:
                ax.plot(
                    [b3_all_coords[ci, 0], target_coord[0]],
                    [b3_all_coords[ci, 1], target_coord[1]],
                    color=color, alpha=0.55, linewidth=0.85,
                    linestyle="--", zorder=2,
                )

        # 5. Unpaired dHCP items (○) — only show 34-68 range
        for _, row in ca_df.iterrows():
            iid = row["item_id"]
            factor = row["assigned_factor"]
            color = _factor_color(factor)
            ci = b3_id_to_idx.get(iid)
            if ci is None:
                continue
            ax.scatter(b3_all_coords[ci, 0], b3_all_coords[ci, 1],
                       c=color, s=85, marker="o", zorder=5,
                       edgecolors="black", linewidths=0.8)
            ax.annotate(iid,
                        (b3_all_coords[ci, 0], b3_all_coords[ci, 1]),
                        fontsize=4.0, ha="center", va="bottom", xytext=(0, 4),
                        textcoords="offset points", color="#444444")

        # 6. EF factor centroids (★)
        for factor, ccoord in ef_coord_map.items():
            color = _factor_color(factor)
            ax.scatter(ccoord[0], ccoord[1], c=color, s=420, marker="*",
                       zorder=7, edgecolors="black", linewidths=0.9)
            ax.annotate(_abbrev(factor), (ccoord[0], ccoord[1]),
                        fontsize=7.5, ha="center", va="bottom", xytext=(0, 10),
                        textcoords="offset points", color="black", fontweight="bold")

        # 7. Single visual UC centroid (◆)
        ax.scatter(visual_uc_coord[0], visual_uc_coord[1],
                   c=FACTOR_COLORS[UNNAMED_CLUSTER], s=380, marker="D",
                   zorder=7, edgecolors="black", linewidths=1.1)
        ax.annotate(uc_abbrev, (visual_uc_coord[0], visual_uc_coord[1]),
                    fontsize=7.5, ha="center", va="bottom", xytext=(0, 10),
                    textcoords="offset points", color="black", fontweight="bold")

        ax.set_title(METHOD_TITLES[method], fontsize=10, fontweight="bold")
        ax.set_xlabel("MDS-1", fontsize=9)
        ax.set_ylabel("MDS-2", fontsize=9)
        ax.tick_params(labelsize=7)

    uc_label = _factor_label(UNNAMED_CLUSTER)
    factor_handles = [
        mpatches.Patch(color=_factor_color(f), label=f"{_abbrev(f)} — {_factor_label(f)}")
        for f in FACTOR_ORDER
    ] + [mpatches.Patch(color=FACTOR_COLORS[UNNAMED_CLUSTER],
                        label=f"{uc_abbrev} — {uc_label}")]
    type_handles = [
        plt.Line2D([0], [0], marker="^", color="gray", markersize=9,
                   linestyle="None", label="B4 paired (expert-guided)"),
        plt.Line2D([0], [0], marker="v", color="gray", markersize=9,
                   linestyle="None", label="B3 paired"),
        plt.Line2D([0], [0], marker="o", color="gray", markersize=8,
                   linestyle="None", label="Unpaired dHCP 34-68 (LOO assigned)"),
        plt.Line2D([0], [0], marker="*", color="gray", markersize=13,
                   linestyle="None", label="Expert-guided cognitive facet centroid"),
        plt.Line2D([0], [0], marker="D",
                   color=FACTOR_COLORS[UNNAMED_CLUSTER], markersize=10,
                   linestyle="None", label=f"Visual {uc_label} centroid (display anchor)"),
        plt.Line2D([0], [0], color="gray", linewidth=1.0, linestyle="--",
                   label="Unpaired → assigned centroid link"),
        plt.Line2D([0], [0], color="gray", linewidth=0.7, linestyle="-",
                   alpha=0.5, label="B4↔B3 pair link (faint)"),
    ]
    axes[-1].legend(handles=factor_handles + type_handles,
                    loc="upper left", bbox_to_anchor=(1.01, 1),
                    fontsize=8, framealpha=0.9)

    fig.suptitle(
        "MDS Overview: B4↔B3 Pairing and LOO Cognitive Facet Assignment — v6_cog\n"
        f"({uc_abbrev} centroid = visual anchor only; actual assignments use LOO centroids)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = outdir / "figures" / "01b_pairing_three_subplots.png"
    fig.savefig(str(out), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved -> {out.name}")


def plot_centroid_similarity_heatmap(
    centroid_assignments: pd.DataFrame,
    method_name: str,
    outdir: Path,
    fig_prefix: str = "03",
    item_scope: str = "dHCP 34-68",
) -> None:
    """
    Heatmap: unpaired items (rows) × all centroids (columns, including UC).
    sim_UC for each row is the LOO similarity used in the assignment decision.
    Assigned centroid marked with ★.
    fig_prefix controls the output filename (e.g. "03" or "03b").
    item_scope is used in the title to describe which items are shown.
    """
    print(f"  [{fig_prefix}] Centroid similarity heatmap ({method_name}, {item_scope})...")

    sim_cols = [c for c in centroid_assignments.columns if c.startswith("sim_")]
    if not sim_cols:
        print("    No similarity columns — skipping.")
        return

    item_ids = centroid_assignments["item_id"].tolist()
    sim_matrix = centroid_assignments[sim_cols].values.astype(float)
    factor_abbrevs = [c.replace("sim_", "") for c in sim_cols]
    assigned_factors = centroid_assignments["assigned_factor"].tolist()
    assigned_abbrevs = [FACTOR_ABBREV.get(f, f) for f in assigned_factors]

    # v6: UC-redirected items belong to best_ef in the model — ★ should mark that column
    if "best_ef_factor" in centroid_assignments.columns:
        model_abbrevs_raw = [
            FACTOR_ABBREV.get(bef, bef)
            if af == UNNAMED_CLUSTER and pd.notna(bef) and bef else FACTOR_ABBREV.get(af, af)
            for af, bef in zip(centroid_assignments["assigned_factor"],
                               centroid_assignments["best_ef_factor"])
        ]
    else:
        model_abbrevs_raw = list(assigned_abbrevs)

    sort_key = list(zip(model_abbrevs_raw, -centroid_assignments["centroid_similarity"].values))
    order = sorted(range(len(item_ids)), key=lambda i: sort_key[i])
    sim_matrix = sim_matrix[order]
    item_ids_s = [item_ids[i] for i in order]
    assigned_s = [assigned_abbrevs[i] for i in order]    # LOO assignment (for display)
    model_s = [model_abbrevs_raw[i] for i in order]       # final model assignment (for ★)

    fig_h = max(5, len(item_ids) * 0.55)
    fig, ax = plt.subplots(figsize=(len(factor_abbrevs) * 2.0, fig_h))
    im = ax.imshow(sim_matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Cosine Similarity")

    uc_abbrev = _abbrev(UNNAMED_CLUSTER)
    for i, (iid, asgn, model_asgn) in enumerate(zip(item_ids_s, assigned_s, model_s)):
        is_uc_redirected = (model_asgn != asgn)
        for j, val in enumerate(sim_matrix[i]):
            star = "★" if factor_abbrevs[j] == model_asgn else ""
            # Mark UC column with * for UC-redirected items (LOO assigned to UC)
            uc_note = "*" if (is_uc_redirected and factor_abbrevs[j] == uc_abbrev) else ""
            color = "white" if val > 0.72 else "#222222"
            ax.text(j, i, f"{star}{val:.2f}{uc_note}", ha="center", va="center",
                    fontsize=11, color=color)

    ax.set_xticks(range(len(factor_abbrevs)))
    ax.set_xticklabels(factor_abbrevs, fontsize=13, fontweight="bold")
    for lbl in ax.get_xticklabels():
        if lbl.get_text() == uc_abbrev:
            lbl.set_color(FACTOR_COLORS[UNNAMED_CLUSTER])

    ax.set_yticks(range(len(item_ids_s)))
    ax.set_yticklabels(
        [f"{iid} → {ms}" for iid, ms in zip(item_ids_s, model_s)],
        fontsize=10, fontfamily="monospace",
    )
    ax.set_title(
        f"Centroid Similarity Heatmap — {method_name} ({item_scope}, v6_cog: ★ = final model assignment)\n"
        f"(sim_{uc_abbrev}* = item was LOO-UC and redirected to best EF; ★ marks the final assigned column)",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    out = outdir / "figures" / f"{fig_prefix}_centroid_sim_heatmap_{method_name}.png"
    fig.savefig(str(out), dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved -> {out.name}")


def plot_assignment_comparison(
    all_centroid_assignments: dict[str, pd.DataFrame],
    outdir: Path,
) -> None:
    """
    Heatmap: rows = unpaired 34-68 items; columns = centroid methods.
    Color = assigned factor (or grey = UC).
    Red border marks disagreements across methods.
    """
    print("  [4/6] Assignment comparison across centroid methods...")

    methods = list(all_centroid_assignments.keys())
    common_items = all_centroid_assignments[methods[0]]["item_id"].tolist()

    factor_order_ext = FACTOR_ORDER + [UNNAMED_CLUSTER]
    abbrev_to_factor = {v: k for k, v in FACTOR_ABBREV.items()}
    factor_to_int = {f: i for i, f in enumerate(factor_order_ext)}

    assign_abbrev: dict[str, dict[str, str]] = {}
    for method, df in all_centroid_assignments.items():
        assign_abbrev[method] = {
            iid: FACTOR_ABBREV.get(factor, factor)
            for iid, factor in zip(df["item_id"], df["assigned_factor"])
        }

    mat = np.full((len(common_items), len(methods)), len(factor_order_ext), dtype=int)
    for j, method in enumerate(methods):
        for i, iid in enumerate(common_items):
            abbrev = assign_abbrev[method].get(iid, "")
            factor = abbrev_to_factor.get(abbrev, "unassigned")
            mat[i, j] = factor_to_int.get(factor, len(factor_order_ext))

    colors_list = (
        [FACTOR_COLORS[f] for f in FACTOR_ORDER]
        + [FACTOR_COLORS[UNNAMED_CLUSTER]]
        + [FACTOR_COLORS["unassigned"]]
    )
    cmap = ListedColormap(colors_list)
    norm = BoundaryNorm(range(len(colors_list) + 1), cmap.N)

    stroke = [mpe.withStroke(linewidth=1.5, foreground="black")]
    fig_h = max(5, len(common_items) * 0.55)
    fig, ax = plt.subplots(figsize=(len(methods) * 2.8, fig_h))
    ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)

    for i, iid in enumerate(common_items):
        for j, method in enumerate(methods):
            label = assign_abbrev[method].get(iid, "?")
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white",
                    path_effects=stroke)
        row_labels = [assign_abbrev[m].get(iid, "?") for m in methods]
        if len(set(row_labels)) > 1:
            ax.add_patch(
                plt.Rectangle((-0.5, i - 0.5), len(methods), 1,
                              fill=False, edgecolor="#DD2222", linewidth=2.0, zorder=5)
            )

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace("_", "\n") for m in methods], fontsize=9)
    ax.set_yticks(range(len(common_items)))
    ax.set_yticklabels(common_items, fontsize=8, fontfamily="monospace")
    uc_abbrev = _abbrev(UNNAMED_CLUSTER)
    uc_label = _factor_label(UNNAMED_CLUSTER)
    ax.set_title(
        "Centroid Method Comparison: Cognitive Facet Assignment of Unpaired B3 Items 34-68 (v6_cog LOO)\n"
        f"(red border = disagreement across methods; grey = {uc_label})",
        fontsize=10, fontweight="bold",
    )
    legend_handles = [
        mpatches.Patch(color=FACTOR_COLORS[f], label=f"{_abbrev(f)} — {_factor_label(f)}")
        for f in FACTOR_ORDER
    ] + [mpatches.Patch(color=FACTOR_COLORS[UNNAMED_CLUSTER],
                        label=f"{uc_abbrev} — {uc_label}")]
    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.01, 1), fontsize=8, framealpha=0.9)
    plt.tight_layout()
    out = outdir / "figures" / "04_centroid_assignment_comparison.png"
    fig.savefig(str(out), dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved -> {out.name}")


def plot_factor_sizes(
    paired_df: pd.DataFrame,
    b3_data_id_set: set[str],
    all_full_assignments: dict[str, pd.DataFrame],
    outdir: Path,
) -> None:
    """
    Bar chart: factor sizes in paired-only vs each expanded model.
    UC bar shows items in UC (LOO-assigned + sparse-EF merged).
    """
    print("  [5/6] Factor size bar chart...")

    factor_order_ext = FACTOR_ORDER + [UNNAMED_CLUSTER]
    abbrevs = [_abbrev(f) for f in factor_order_ext]
    bar_colors = [_factor_color(f) for f in factor_order_ext]

    paired_in_data = paired_df[paired_df["b3_item_id"].isin(b3_data_id_set)]
    paired_counts = (
        paired_in_data.groupby("b4_factor").size()
        .reindex(FACTOR_ORDER, fill_value=0)
    )
    paired_counts_ext = list(paired_counts.values) + [0]

    n_panels = 1 + len(all_full_assignments)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 6), sharey=True)

    ax0 = axes[0]
    ax0.bar(abbrevs, paired_counts_ext, color=bar_colors, alpha=0.85, edgecolor="white")
    ax0.set_title("Paired-only\n(34-68 items)", fontsize=10, fontweight="bold")
    ax0.set_ylabel("N items")
    for i, v in enumerate(paired_counts_ext):
        if v:
            ax0.text(i, v + 0.05, str(int(v)), ha="center", va="bottom", fontsize=9)

    uc_label = _factor_label(UNNAMED_CLUSTER)
    for ax, (method, full_df) in zip(axes[1:], all_full_assignments.items()):
        counts = (
            full_df.dropna(subset=["assigned_factor"])
            .groupby("assigned_factor").size()
            .reindex(factor_order_ext, fill_value=0)
        )
        ax.bar(abbrevs, counts.values, color=bar_colors, alpha=0.85, edgecolor="white")
        ax.set_title(f"Expanded\n({method.replace('_', ' ')})",
                     fontsize=10, fontweight="bold")
        for i, v in enumerate(counts.values):
            if v:
                ax.text(i, v + 0.05, str(int(v)), ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        f"B3 Items 34-68: Cognitive Facet Sizes — Paired-only vs Expanded Models (v6_cog LOO)\n"
        f"({_abbrev(UNNAMED_CLUSTER)} = {uc_label}: LOO-UC redirected + sparse-EF excluded)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = outdir / "figures" / "05_factor_sizes.png"
    fig.savefig(str(out), dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved -> {out.name}")


def plot_uc_similarity_profile(
    uc_profiles: dict[str, pd.DataFrame],
    outdir: Path,
) -> None:
    """
    Violin + strip plot: distribution of similarity margins for UC items
    across centroid methods.

    margin = sim_UC − best_EF_sim
      positive ⇒ UC was genuinely nearest (LOO-assigned)
      negative ⇒ item preferred an EF factor (sparse-EF merge)

    Each method is a column; items colored by source (loo_assigned / sparse_ef).
    """
    print("  [6/6] UC similarity profile plot...")

    methods = [m for m in CENTROID_METHODS if m in uc_profiles and not uc_profiles[m].empty]
    if not methods:
        print("    No UC items — skipping UC similarity profile plot.")
        return

    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 6), sharey=True)
    if n_methods == 1:
        axes = [axes]

    source_colors = {"loo_uc_redirected": "#4878CF", "sparse_ef_excluded": "#E8705A"}

    for ax, method in zip(axes, methods):
        df = uc_profiles[method]
        if df.empty:
            ax.text(0.5, 0.5, "No UC-diagnostic items", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(method.replace("_", "\n"), fontsize=10)
            continue

        # Violin per source group (only for groups with 2+ valid numeric values)
        groups = sorted(df["source"].unique())
        positions = list(range(len(groups)))
        data_by_group = [df.loc[df["source"] == g, "margin"].dropna().values for g in groups]

        for i, (group_data, g) in enumerate(zip(data_by_group, groups)):
            if len(group_data) >= 2:
                parts = ax.violinplot([group_data], positions=[i],
                                      showmedians=True, showextrema=True)
                for patch in parts["bodies"]:
                    patch.set_facecolor(source_colors.get(g, "#888888"))
                    patch.set_alpha(0.55)
                parts["cmedians"].set_color("black")
                parts["cbars"].set_color("black")
                parts["cmins"].set_color("black")
                parts["cmaxes"].set_color("black")
            elif len(group_data) == 1:
                # Single point — draw a horizontal line instead
                ax.axhline(group_data[0], xmin=(i / len(groups)) + 0.05,
                           xmax=((i + 1) / len(groups)) - 0.05,
                           color=source_colors.get(g, "#888888"),
                           linewidth=2, alpha=0.7)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels([g.replace("_", "\n") for g in groups], fontsize=9)
        ax.set_title(method.replace("_", "\n"), fontsize=10, fontweight="bold")
        ax.set_ylabel("Margin (sim_UC − best_EF_sim)" if ax == axes[0] else "")
        ax.tick_params(labelsize=8)

        # Annotate item IDs — use per-group DataFrames to keep NaN-margin items
        rng2 = np.random.default_rng(42)
        for i, g in enumerate(groups):
            group_df = df[df["source"] == g].reset_index(drop=True)
            color = source_colors.get(g, "#888888")
            for row in group_df.itertuples():
                margin_val = row.margin
                if np.isnan(margin_val):
                    # Sparse-EF items with no margin: show at y=0 with an "x" marker
                    jitter_val = rng2.uniform(-0.08, 0.08)
                    ax.scatter(i + jitter_val, 0, s=60, color=color, marker="x",
                               linewidths=1.5, zorder=4, alpha=0.7)
                    ax.annotate(row.item_id, (i + jitter_val, 0),
                                fontsize=5.5, ha="center", va="bottom",
                                xytext=(0, 4), textcoords="offset points",
                                color="#555555")
                else:
                    jitter_val = rng2.uniform(-0.08, 0.08)
                    ax.scatter(i + jitter_val, margin_val, s=40, alpha=0.8,
                               color=color, edgecolors="white", linewidths=0.4, zorder=3)
                    ax.annotate(row.item_id, (i + jitter_val, margin_val),
                                fontsize=5.5, ha="center", va="bottom",
                                xytext=(0, 3), textcoords="offset points",
                                color="#333333")

    legend_handles = [
        mpatches.Patch(color=source_colors["loo_uc_redirected"],
                       label="LOO-UC redirected (reassigned to best cognitive facet)"),
        mpatches.Patch(color=source_colors["sparse_ef_excluded"],
                       label="Sparse facet excluded (< 3 items in model)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), fontsize=9, framealpha=0.9)
    fig.suptitle(
        f"UC Similarity Margin: sim_UC − best_EF_sim (v6_cog)\n"
        "(positive = UC was nearest; items shown are excluded from model/redirected)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out = outdir / "figures" / "06_uc_similarity_profile.png"
    fig.savefig(str(out), dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved -> {out.name}")


# ============================================================================
# NEW FIGURE: 01c — dHCP 34-68 B3 items, UC redirected with connections
# ============================================================================


def plot_b3_dhcp_uc_redirected(
    b3_items: pd.DataFrame,
    b3_emb: np.ndarray,
    paired_df: pd.DataFrame,
    b3_data_id_set: set[str],
    ef_centroids: dict[str, np.ndarray],
    centroid_assignments_34_68: pd.DataFrame,
    visual_unnamed_centroid: np.ndarray,
    outdir: Path,
    fig_name: str = "01c_b3_dhcp_uc_redirected.png",
    ca_all_df: pd.DataFrame | None = None,
) -> None:
    """
    MDS plot showing ALL 91 B3 items colored by expert-guided cognitive facet.

    dHCP 34-68 items are drawn larger; non-dHCP items smaller and faded.
    Connection lines drawn ONLY for UNPAIRED dHCP items:
      - LOO-EF-assigned: thin solid line to EF centroid
      - UC-redirected: dashed line to best_ef centroid; thinner grey dashed
        to 2nd-best EF if best_ef_sim and second_ef_sim are within TIE_THRESHOLD.
    Uncertain items (uc_margin < UC_UNCERTAINTY_MARGIN): hollow marker + '?'.
    ca_all_df: full LOO assignments for all 91 B3 items (used to color non-dHCP items).
    """
    print(f"  [new] {fig_name}...")

    b3_id_to_idx = {row["item_id"]: i for i, row in b3_items.iterrows()}
    paired_map = dict(zip(paired_df["b3_item_id"], paired_df["b4_factor"]))
    factors_available = [f for f in FACTOR_ORDER if f in ef_centroids]

    # Build LOO assignment lookups
    ca_lookup = centroid_assignments_34_68.set_index("item_id")
    ca_all_lookup = ca_all_df.set_index("item_id") if ca_all_df is not None else None

    # ---- Joint MDS embedding ----
    # [B3 all | EF centroids | visual_UC]
    ef_centroid_stack = np.vstack([ef_centroids[f] for f in factors_available])
    combined = np.vstack([b3_emb, ef_centroid_stack,
                          visual_unnamed_centroid.reshape(1, -1)])
    dist = np.clip(1 - cosine_similarity(combined), 0, None)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=4)
    coords = mds.fit_transform(dist)

    n_b3 = len(b3_items)
    b3_coords = coords[:n_b3]
    ef_centroid_coords = {f: coords[n_b3 + fi] for fi, f in enumerate(factors_available)}

    dhcp_ids = set(b3_data_id_set)

    def _final_factor(iid: str) -> tuple[str | None, bool, bool]:
        """Return (final_factor, is_uc_redirected, is_uncertain)."""
        if iid in paired_map:
            return paired_map[iid], False, False
        # dHCP unpaired
        if iid in ca_lookup.index:
            row = ca_lookup.loc[iid]
            af = row.get("assigned_factor", None)
            if af == UNNAMED_CLUSTER:
                bef = row.get("best_ef_factor", None)
                margin_val = row.get("uc_margin", float("nan"))
                # Flag as uncertain when UC strongly dominates (margin > threshold)
                # — consistent with figure 07 red highlighting logic
                uncertain = (not np.isnan(float(margin_val))
                             and float(margin_val) > UC_UNCERTAINTY_MARGIN)
                return bef if pd.notna(bef) else None, True, uncertain
            return af, False, False
        # non-dHCP
        if ca_all_lookup is not None and iid in ca_all_lookup.index:
            row = ca_all_lookup.loc[iid]
            af = row.get("assigned_factor", None)
            if af == UNNAMED_CLUSTER:
                bef = row.get("best_ef_factor", None)
                return bef if pd.notna(bef) else None, True, False
            return af, False, False
        return None, False, False

    fig, ax = plt.subplots(figsize=(16, 12))

    # dHCP items larger, non-dHCP smaller and faded
    s_dhcp, s_non_dhcp = 130, 55
    fs_dhcp, fs_non_dhcp = 5.0, 3.5
    alpha_non_dhcp = 0.55

    # ---- Connection lines (ONLY for unpaired dHCP items) ----
    for i, row in b3_items.iterrows():
        iid = row["item_id"]
        if iid not in dhcp_ids or iid in paired_map:
            continue  # skip paired and non-dHCP
        if iid not in ca_lookup.index:
            continue
        ca_row = ca_lookup.loc[iid]
        assigned = ca_row.get("assigned_factor", None)

        if assigned == UNNAMED_CLUSTER:
            best_ef = ca_row.get("best_ef_factor", None)
            best_ef_sim_val = float(ca_row.get("best_ef_sim", float("nan")))
            if pd.notna(best_ef) and best_ef in ef_centroid_coords:
                cx, cy = b3_coords[i]
                tx, ty = ef_centroid_coords[best_ef]
                ax.plot([cx, tx], [cy, ty],
                        color=_factor_color(best_ef), alpha=0.65,
                        linewidth=1.2, linestyle="--", zorder=2)
            # Draw 2nd line if best_ef and second_ef are within TIE_THRESHOLD
            s2 = float(ca_row.get("second_ef_sim", float("nan")))
            second_ef = ca_row.get("second_ef_factor", None)
            is_close_2nd = (not np.isnan(best_ef_sim_val) and not np.isnan(s2)
                            and best_ef_sim_val - s2 < TIE_THRESHOLD)
            if is_close_2nd and pd.notna(second_ef) and second_ef in ef_centroid_coords:
                cx2, cy2 = b3_coords[i]
                tx2, ty2 = ef_centroid_coords[second_ef]
                ax.plot([cx2, tx2], [cy2, ty2],
                        color="#AAAAAA", alpha=0.5,
                        linewidth=0.55, linestyle="--", zorder=2)
        else:
            if pd.notna(assigned) and assigned in ef_centroid_coords:
                cx, cy = b3_coords[i]
                tx, ty = ef_centroid_coords[assigned]
                ax.plot([cx, tx], [cy, ty],
                        color=_factor_color(assigned), alpha=0.35,
                        linewidth=0.7, linestyle="-", zorder=2)

    # ---- Draw all B3 items ----
    for i, row in b3_items.iterrows():
        iid = row["item_id"]
        is_dhcp = iid in dhcp_ids
        final_f, is_uc_redir, is_uncertain = _final_factor(iid)
        color = _factor_color(final_f) if final_f else "#AAAAAA"
        x, y = b3_coords[i]
        s = s_dhcp if is_dhcp else s_non_dhcp
        fs = fs_dhcp if is_dhcp else fs_non_dhcp
        alpha = 1.0 if is_dhcp else alpha_non_dhcp

        if iid in paired_map:
            # Paired (▼)
            ax.scatter(x, y, c=color, s=s, marker="v", zorder=5,
                       edgecolors="white", linewidths=0.7, alpha=alpha)
            ax.annotate(iid, (x, y), fontsize=fs, ha="center", va="top",
                        xytext=(0, -4), textcoords="offset points", color="#333333")
        elif is_dhcp:
            # Unpaired dHCP (○)
            if is_uncertain:
                ax.scatter(x, y, c="none", s=s, marker="o", zorder=5,
                           edgecolors=color, linewidths=1.8)
                ax.annotate(f"{iid} ?", (x, y), fontsize=fs, ha="center", va="bottom",
                            xytext=(0, 5), textcoords="offset points",
                            color=color, fontweight="bold")
            else:
                ax.scatter(x, y, c=color, s=s, marker="o", zorder=5,
                           edgecolors="white", linewidths=0.7)
                ax.annotate(iid, (x, y), fontsize=fs, ha="center", va="bottom",
                            xytext=(0, 4), textcoords="offset points", color="#333333")
        else:
            # Non-dHCP (○ smaller)
            ax.scatter(x, y, c=color, s=s, marker="o", zorder=3,
                       edgecolors="white", linewidths=0.5, alpha=alpha)
            ax.annotate(iid, (x, y), fontsize=fs, ha="center", va="bottom",
                        xytext=(0, 3), textcoords="offset points", color="#666666",
                        alpha=alpha)

    # ---- EF facet centroids (★) ----
    for factor, ccoord in ef_centroid_coords.items():
        color = _factor_color(factor)
        ax.scatter(ccoord[0], ccoord[1], c=color, s=500, marker="*",
                   zorder=8, edgecolors="black", linewidths=0.9)
        ax.annotate(_abbrev(factor), (ccoord[0], ccoord[1]),
                    fontsize=9, ha="center", va="bottom", xytext=(0, 12),
                    textcoords="offset points", color="black", fontweight="bold")

    # ---- Legend ----
    factor_handles = [
        mpatches.Patch(color=_factor_color(f), label=f"{_abbrev(f)} — {_factor_label(f)}")
        for f in FACTOR_ORDER
    ]
    type_handles = [
        plt.Line2D([0], [0], marker="v", color="gray", markersize=9,
                   linestyle="None", label="Paired B3 item (expert-guided)"),
        plt.Line2D([0], [0], marker="o", color="gray", markersize=8,
                   linestyle="None", label="LOO cognitive facet-assigned item (unpaired dHCP)"),
        plt.Line2D([0], [0], marker="o", color="none", markeredgecolor="#888888",
                   markeredgewidth=1.8, markersize=9,
                   linestyle="None", label=f"LOO uncertain (uc_margin < {UC_UNCERTAINTY_MARGIN:.2f}): ?"),
        plt.Line2D([0], [0], color="gray", linewidth=1.2, linestyle="--",
                   label="UC-redirected → best cognitive facet"),
        plt.Line2D([0], [0], color="#AAAAAA", linewidth=0.55, linestyle="--",
                   label=f"Ambiguous 2nd facet (Δ < {TIE_THRESHOLD:.3f})"),
        plt.Line2D([0], [0], color="gray", linewidth=0.7, linestyle="-",
                   alpha=0.35, label="EF-assigned → facet centroid"),
        plt.Line2D([0], [0], marker="*", color="gray", markersize=12,
                   linestyle="None", label="Expert-guided cognitive facet centroid"),
        plt.Line2D([0], [0], marker="o", color="#AAAAAA", markersize=6,
                   linestyle="None", alpha=0.6, label="B3 item (no dHCP data)"),
    ]
    ax.legend(handles=factor_handles + type_handles,
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, framealpha=0.9)

    ax.set_title(
        f"B3 All 91 Items: Expert-Guided Cognitive Facet Assignment (v6_cog, mixed centroids)\n"
        f"(▼ = paired; ○ = LOO-assigned; ? = uncertain; dashed = UC-redirected connections [unpaired dHCP])",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("MDS-1", fontsize=9)
    ax.set_ylabel("MDS-2", fontsize=9)
    plt.tight_layout()
    out = outdir / "figures" / fig_name
    fig.savefig(str(out), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved -> {out.name}")


# ============================================================================
# NEW FIGURE: 07 — UC Decision Logic (lollipop + scatter)
# ============================================================================


def plot_uc_decision_logic(
    centroid_assignments_34_68: dict[str, pd.DataFrame],
    uc_profiles: dict[str, pd.DataFrame],
    outdir: Path,
    method: str = "mixed_centroids",
) -> None:
    """
    Figure 07: Two sub-panels explaining the UC reassignment logic.

    Sub-panel A (lollipop):
      - Rows = UC-redirected items sorted by uc_margin descending
      - One dot per EF cognitive facet at its similarity value, colored by facet
      - Vertical highlight line for best_ef_sim (reassignment target)
      - Horizontal bracket/annotation showing uc_margin and sim_UC
      - Red row background for uncertain items (margin < UC_UNCERTAINTY_MARGIN)

    Sub-panel B (scatter):
      - All LOO-assigned items: x = sim_UC, y = sim_best_ef
      - Items above diagonal: UC was nearest (grey = UC-redirected in model)
      - Items below/on diagonal: EF was nearest (colored by assigned facet)
      - Diagonal line x = y; red shaded zone = strong UC preference (margin > threshold)
      - All items labeled with item_id
    """
    print("  [new] 07_uc_decision_logic.png...")

    if method not in centroid_assignments_34_68:
        print(f"    Method {method!r} not available — skipping 07.")
        return

    ca_df = centroid_assignments_34_68[method]
    uc_prof = uc_profiles.get(method, pd.DataFrame())

    uc_redirected = uc_prof[uc_prof["source"] == "loo_uc_redirected"].copy()
    if uc_redirected.empty:
        print("    No UC-redirected items — skipping 07.")
        return

    uc_redirected = uc_redirected.sort_values("margin", ascending=False).reset_index(drop=True)

    factors_available = [f for f in FACTOR_ORDER if f"sim_{_abbrev(f)}" in ca_df.columns]
    uc_abbrev = _abbrev(UNNAMED_CLUSTER)

    fig, (ax_lollipop, ax_scatter) = plt.subplots(
        1, 2, figsize=(20, max(8, len(uc_redirected) * 0.7 + 3))
    )

    # ---- Sub-panel A: Lollipop ----
    ax = ax_lollipop
    ca_lookup = ca_df.set_index("item_id")

    for row_idx, item_row in uc_redirected.iterrows():
        iid = item_row["item_id"]
        margin_val = item_row["margin"]
        strong_uc = (not np.isnan(float(margin_val))
                     and float(margin_val) > UC_UNCERTAINTY_MARGIN)

        # Red background band for items where UC strongly dominates best EF
        if strong_uc:
            ax.axhspan(row_idx - 0.45, row_idx + 0.45, color="#FFEEEE", zorder=0)

        # Horizontal grey baseline
        ax.axhline(row_idx, color="#DDDDDD", linewidth=0.8, zorder=1)

        if iid not in ca_lookup.index:
            continue
        ca_row = ca_lookup.loc[iid]
        best_ef = item_row["best_ef_factor"]
        best_ef_sim = item_row["best_ef_sim"]
        uc_sim = item_row["uc_sim"]

        # Dots for each EF facet
        for f in factors_available:
            col = f"sim_{_abbrev(f)}"
            if col in ca_row.index:
                sim_val = float(ca_row[col])
                color = _factor_color(f)
                size = 120 if f == best_ef else 60
                zorder = 5 if f == best_ef else 3
                ax.scatter(sim_val, row_idx, c=color, s=size, zorder=zorder,
                           edgecolors="black" if f == best_ef else "none",
                           linewidths=0.8)

        # UC similarity dot (grey)
        if not np.isnan(float(uc_sim)):
            ax.scatter(float(uc_sim), row_idx,
                       c=FACTOR_COLORS[UNNAMED_CLUSTER], s=80, marker="D",
                       zorder=4, edgecolors="black", linewidths=0.6)

        # Vertical line at best_ef_sim
        if not np.isnan(float(best_ef_sim)):
            ax.axvline(float(best_ef_sim), color=_factor_color(best_ef),
                       alpha=0.25, linewidth=1.0, zorder=1)

        # Bracket annotation: uc_margin
        if not np.isnan(float(margin_val)) and not np.isnan(float(best_ef_sim)):
            x_low = min(float(uc_sim), float(best_ef_sim))
            x_high = max(float(uc_sim), float(best_ef_sim))
            y_bracket = row_idx + 0.30
            ax.annotate(
                "", xy=(x_high, y_bracket), xytext=(x_low, y_bracket),
                arrowprops=dict(arrowstyle="<->", color="#555555", lw=0.9),
            )
            ax.text((x_low + x_high) / 2, y_bracket + 0.12,
                    f"Δ={float(margin_val):+.3f}", ha="center", va="bottom",
                    fontsize=6.5, color="#555555")

    ax.set_yticks(range(len(uc_redirected)))
    ax.set_yticklabels(
        [f"{'!' if (not np.isnan(float(r.margin)) and float(r.margin) > UC_UNCERTAINTY_MARGIN) else ' '} "
         f"{r.item_id}" for r in uc_redirected.itertuples()],
        fontsize=8, fontfamily="monospace",
    )
    ax.set_xlabel("Cosine Similarity", fontsize=10)
    ax.set_xlim(0.2, 1.0)
    ax.set_title(
        f"UC-Redirected Items: All Cognitive Facet Similarities\n"
        f"({method.replace('_', ' ')} — sorted by uc_margin desc; "
        f"red row = strong UC preference [Δ > {UC_UNCERTAINTY_MARGIN:.2f}])",
        fontsize=10, fontweight="bold",
    )
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    # Facet legend for lollipop
    lollipop_handles = [
        mpatches.Patch(color=_factor_color(f), label=f"{_abbrev(f)} — {_factor_label(f)}")
        for f in factors_available
    ] + [
        mpatches.Patch(color=FACTOR_COLORS[UNNAMED_CLUSTER], label=f"{uc_abbrev} (LOO UC centroid)"),
        mpatches.Patch(color="#FFEEEE", label=f"Strong UC preference (Δ > {UC_UNCERTAINTY_MARGIN:.2f})", alpha=0.7),
    ]
    ax.legend(handles=lollipop_handles, loc="lower right", fontsize=7.5, framealpha=0.9)

    # ---- Sub-panel B: Scatter ----
    ax2 = ax_scatter

    # Diagonal
    lo, hi = 0.25, 1.0
    ax2.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--",
             zorder=1, label="x = y (UC = EF)")

    # Red band: sim_UC - sim_best_ef > UC_UNCERTAINTY_MARGIN (UC strongly dominates)
    x_shade = np.linspace(lo, hi, 200)
    ax2.fill_between(x_shade,
                     lo,
                     np.maximum(lo, x_shade - UC_UNCERTAINTY_MARGIN),
                     alpha=0.15, color="#FFCCCC", label=f"Strong UC preference (Δ > {UC_UNCERTAINTY_MARGIN:.2f})")

    rng = np.random.default_rng(99)
    for _, ca_row in ca_df.iterrows():
        iid = ca_row["item_id"]
        uc_sim = float(ca_row.get(f"sim_{uc_abbrev}", float("nan")))
        assigned = ca_row.get("assigned_factor", None)
        best_ef_f = ca_row.get("best_ef_factor", None)
        best_ef_s = float(ca_row.get("best_ef_sim", float("nan")))

        if np.isnan(uc_sim) or np.isnan(best_ef_s):
            continue

        if assigned == UNNAMED_CLUSTER:
            # UC-redirected in model
            color = FACTOR_COLORS[UNNAMED_CLUSTER]
            marker = "D"
            size = 90
        else:
            color = _factor_color(assigned) if pd.notna(assigned) else "#AAAAAA"
            marker = "o"
            size = 70

        jx = rng.uniform(-0.003, 0.003)
        jy = rng.uniform(-0.003, 0.003)
        ax2.scatter(uc_sim + jx, best_ef_s + jy, c=color, s=size,
                    marker=marker, zorder=4,
                    edgecolors="black" if assigned == UNNAMED_CLUSTER else "none",
                    linewidths=0.7, alpha=0.85)
        ax2.annotate(iid, (uc_sim + jx, best_ef_s + jy),
                     fontsize=5.5, ha="center", va="bottom", xytext=(0, 4),
                     textcoords="offset points", color="#333333")

    ax2.set_xlabel("sim_UC (LOO)", fontsize=10)
    ax2.set_ylabel("sim_best_EF", fontsize=10)
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    ax2.set_aspect("equal")
    ax2.set_title(
        f"LOO-Assigned Items: sim_UC vs sim_best_EF\n"
        f"({method.replace('_', ' ')} — ◆ = UC-redirected; ○ = EF-assigned; shaded = uncertainty)",
        fontsize=10, fontweight="bold",
    )
    ax2.grid(linestyle=":", alpha=0.35)

    scatter_handles = [
        mpatches.Patch(color=_factor_color(f), label=f"{_abbrev(f)} — {_factor_label(f)}")
        for f in factors_available
    ] + [
        mpatches.Patch(color=FACTOR_COLORS[UNNAMED_CLUSTER],
                       label=f"{uc_abbrev} redirected (above diagonal)"),
    ]
    ax2.legend(handles=scatter_handles, loc="upper left", fontsize=7.5, framealpha=0.9)

    fig.suptitle(
        "UC Decision Logic — v6_cog Expert-Guided Cognitive Facet Reassignment\n"
        "(Left: per-item facet similarities for UC-redirected items; "
        "Right: sim_UC vs sim_best_EF for all LOO items)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = outdir / "figures" / "07_uc_decision_logic.png"
    fig.savefig(str(out), dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved -> {out.name}")


def plot_uc_decision_logic_all(
    all_ca_all: dict[str, pd.DataFrame],
    b3_data_id_set: set[str],
    outdir: Path,
    method: str = "mixed_centroids",
) -> None:
    """
    Figure 07b: UC decision logic for ALL 52 unpaired B3 items (dHCP + non-dHCP).

    Mirrors figure 07 but sources from the full ca_all DataFrame.
    dHCP items highlighted with a blue row band and [dHCP] label suffix.
    Sub-panel A (lollipop): all UC-redirected items, sorted by uc_margin desc.
    Sub-panel B (scatter):  all 52 items; △/◆ = dHCP, ○/◇ = non-dHCP.
    """
    print("  [new] 07b_uc_decision_logic_all.png...")

    if method not in all_ca_all:
        print(f"    Method {method!r} not available — skipping 07b.")
        return

    ca_df = all_ca_all[method]
    uc_abbrev = _abbrev(UNNAMED_CLUSTER)
    factors_available = [f for f in FACTOR_ORDER if f"sim_{_abbrev(f)}" in ca_df.columns]

    uc_redirected = (
        ca_df[ca_df["assigned_factor"] == UNNAMED_CLUSTER]
        .copy()
        .sort_values("uc_margin", ascending=False)
        .reset_index(drop=True)
    )

    if uc_redirected.empty:
        print("    No UC-redirected items across all 52 unpaired B3 — skipping 07b.")
        return

    fig, (ax_lollipop, ax_scatter) = plt.subplots(
        1, 2, figsize=(22, max(8, len(uc_redirected) * 0.7 + 3))
    )

    # ---- Sub-panel A: Lollipop ----
    ax = ax_lollipop

    for row_idx, item_row in uc_redirected.iterrows():
        iid = item_row["item_id"]
        is_dhcp = iid in b3_data_id_set
        margin_val = float(item_row.get("uc_margin", float("nan")))
        strong_uc = not np.isnan(margin_val) and margin_val > UC_UNCERTAINTY_MARGIN

        if strong_uc:
            ax.axhspan(row_idx - 0.45, row_idx + 0.45, color="#FFEEEE", zorder=0)
        elif is_dhcp:
            ax.axhspan(row_idx - 0.45, row_idx + 0.45, color="#EEF2FF", zorder=0)

        ax.axhline(row_idx, color="#DDDDDD", linewidth=0.8, zorder=1)

        best_ef = item_row.get("best_ef_factor", None)
        best_ef_sim = float(item_row.get("best_ef_sim", float("nan")))
        uc_sim = float(item_row.get(f"sim_{uc_abbrev}", float("nan")))

        for f in factors_available:
            col = f"sim_{_abbrev(f)}"
            if col in item_row.index:
                sim_val = float(item_row[col])
                color = _factor_color(f)
                size = 120 if f == best_ef else 60
                zorder = 5 if f == best_ef else 3
                ax.scatter(sim_val, row_idx, c=color, s=size, zorder=zorder,
                           edgecolors="black" if f == best_ef else "none",
                           linewidths=0.8)

        if not np.isnan(uc_sim):
            ax.scatter(uc_sim, row_idx,
                       c=FACTOR_COLORS[UNNAMED_CLUSTER], s=80, marker="D",
                       zorder=4, edgecolors="black", linewidths=0.6)

        if not np.isnan(best_ef_sim) and pd.notna(best_ef):
            ax.axvline(best_ef_sim, color=_factor_color(best_ef),
                       alpha=0.25, linewidth=1.0, zorder=1)

        if not np.isnan(margin_val) and not np.isnan(best_ef_sim):
            x_low = min(uc_sim, best_ef_sim)
            x_high = max(uc_sim, best_ef_sim)
            y_bracket = row_idx + 0.30
            ax.annotate(
                "", xy=(x_high, y_bracket), xytext=(x_low, y_bracket),
                arrowprops=dict(arrowstyle="<->", color="#555555", lw=0.9),
            )
            ax.text((x_low + x_high) / 2, y_bracket + 0.12,
                    f"Δ={margin_val:+.3f}", ha="center", va="bottom",
                    fontsize=6.5, color="#555555")

    ytick_labels = []
    for r in uc_redirected.itertuples():
        is_dhcp = r.item_id in b3_data_id_set
        margin_val = float(r.uc_margin) if not np.isnan(float(r.uc_margin)) else 0.0
        prefix = "! " if margin_val > UC_UNCERTAINTY_MARGIN else "  "
        suffix = " [dHCP]" if is_dhcp else ""
        ytick_labels.append(f"{prefix}{r.item_id}{suffix}")

    ax.set_yticks(range(len(uc_redirected)))
    ax.set_yticklabels(ytick_labels, fontsize=8, fontfamily="monospace")
    ax.set_xlabel("Cosine Similarity", fontsize=10)
    ax.set_xlim(0.2, 1.0)
    ax.set_title(
        f"UC-Redirected Items: All Cognitive Facet Similarities (all 52 unpaired)\n"
        f"({method.replace('_', ' ')} — sorted by uc_margin desc; "
        f"red = strong UC preference [Δ > {UC_UNCERTAINTY_MARGIN:.2f}]; blue = dHCP 34-68)",
        fontsize=10, fontweight="bold",
    )
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    lollipop_handles = [
        mpatches.Patch(color=_factor_color(f), label=f"{_abbrev(f)} — {_factor_label(f)}")
        for f in factors_available
    ] + [
        mpatches.Patch(color=FACTOR_COLORS[UNNAMED_CLUSTER], label=f"{uc_abbrev} (LOO UC centroid)"),
        mpatches.Patch(color="#FFEEEE", label=f"Strong UC preference [Δ > {UC_UNCERTAINTY_MARGIN:.2f}]", alpha=0.7),
        mpatches.Patch(color="#EEF2FF", label="dHCP item (34-68)", alpha=0.7),
    ]
    ax.legend(handles=lollipop_handles, loc="lower right", fontsize=7.5, framealpha=0.9)

    # ---- Sub-panel B: Scatter (all 52 items) ----
    ax2 = ax_scatter
    lo, hi = 0.25, 1.0
    ax2.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--",
             zorder=1)
    x_shade = np.linspace(lo, hi, 200)
    ax2.fill_between(x_shade,
                     lo,
                     np.maximum(lo, x_shade - UC_UNCERTAINTY_MARGIN),
                     alpha=0.15, color="#FFCCCC")

    rng = np.random.default_rng(99)
    for _, ca_row in ca_df.iterrows():
        iid = ca_row["item_id"]
        is_dhcp = iid in b3_data_id_set
        uc_sim = float(ca_row.get(f"sim_{uc_abbrev}", float("nan")))
        assigned = ca_row.get("assigned_factor", None)
        best_ef_s = float(ca_row.get("best_ef_sim", float("nan")))

        if np.isnan(uc_sim) or np.isnan(best_ef_s):
            continue

        if assigned == UNNAMED_CLUSTER:
            color = FACTOR_COLORS[UNNAMED_CLUSTER]
            marker = "D" if is_dhcp else "d"
            size = 90 if is_dhcp else 65
        else:
            color = _factor_color(assigned) if pd.notna(assigned) else "#AAAAAA"
            marker = "^" if is_dhcp else "o"
            size = 75 if is_dhcp else 50

        jx = rng.uniform(-0.003, 0.003)
        jy = rng.uniform(-0.003, 0.003)
        ax2.scatter(uc_sim + jx, best_ef_s + jy, c=color, s=size,
                    marker=marker, zorder=4,
                    edgecolors="black" if is_dhcp else "none",
                    linewidths=0.7, alpha=0.85)
        ax2.annotate(iid, (uc_sim + jx, best_ef_s + jy),
                     fontsize=5.5, ha="center", va="bottom", xytext=(0, 4),
                     textcoords="offset points", color="#333333")

    ax2.set_xlabel("sim_UC (LOO)", fontsize=10)
    ax2.set_ylabel("sim_best_EF", fontsize=10)
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    ax2.set_aspect("equal")
    ax2.set_title(
        f"ALL 52 Unpaired B3 Items: sim_UC vs sim_best_EF\n"
        f"({method.replace('_', ' ')} — △/◆ = dHCP; ○/◇ = non-dHCP; grey shaded = uncertainty band)",
        fontsize=10, fontweight="bold",
    )
    ax2.grid(linestyle=":", alpha=0.35)

    scatter_handles = [
        mpatches.Patch(color=_factor_color(f), label=f"{_abbrev(f)} — {_factor_label(f)}")
        for f in factors_available
    ] + [
        mpatches.Patch(color=FACTOR_COLORS[UNNAMED_CLUSTER],
                       label=f"{uc_abbrev} redirected"),
        plt.Line2D([0], [0], marker="^", color="gray", markersize=8,
                   markeredgecolor="black", markeredgewidth=0.7,
                   linestyle="None", label="dHCP item (34-68) EF-assigned"),
        plt.Line2D([0], [0], marker="D", color=FACTOR_COLORS[UNNAMED_CLUSTER],
                   markersize=8, markeredgecolor="black", markeredgewidth=0.7,
                   linestyle="None", label="dHCP item (34-68) UC-redirected"),
        plt.Line2D([0], [0], marker="o", color="gray", markersize=7,
                   linestyle="None", label="non-dHCP item EF-assigned"),
        plt.Line2D([0], [0], marker="d", color=FACTOR_COLORS[UNNAMED_CLUSTER],
                   markersize=7, linestyle="None", label="non-dHCP item UC-redirected"),
    ]
    ax2.legend(handles=scatter_handles, loc="upper left", fontsize=7.5, framealpha=0.9)

    fig.suptitle(
        "UC Decision Logic — All 52 Unpaired B3 Items (v6_cog Expert-Guided Cognitive Facet Reassignment)\n"
        "(Left: per-item facet similarities for UC-redirected; "
        "Right: sim_UC vs sim_best_EF for all 52 unpaired items)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = outdir / "figures" / "07b_uc_decision_logic_all.png"
    fig.savefig(str(out), dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved -> {out.name}")


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    print(f"\n{'=' * 70}")
    print("B3 EXPERT-GUIDED MAPPING PIPELINE — v6_cog (UC Reassignment)")
    print(f"{'=' * 70}\n")
    print(f"Model                   : {MODEL}")
    print(f"Output                  : {OUTDIR}")
    print(f"Min items per EF factor : {MIN_ITEMS_PER_EF_FACTOR}\n")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "figures").mkdir(exist_ok=True)
    for method in CENTROID_METHODS:
        (OUTDIR / method).mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Theoretical assignments (returns first-wins + all-factors)
    # ------------------------------------------------------------------
    print(f"{'=' * 70}")
    print("[1/10] Loading theoretical B4 factor assignments...")
    item_to_factor, item_to_all_factors = load_theoretical_assignments()
    print(f"      {len(item_to_factor)} B4 items with theoretical assignments")
    for factor in FACTOR_ORDER:
        n_primary = sum(1 for f in item_to_factor.values() if f == factor)
        n_total_incl_cross = sum(1 for fs in item_to_all_factors.values() if factor in fs)
        print(f"      {_abbrev(factor):6s}: {n_primary} primary  "
              f"{n_total_incl_cross} total (incl. cross-loaders)")

    # ------------------------------------------------------------------
    # Steps 2-3: Embed
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("[2-3/10] Loading model and embedding items...")
    model = load_model(MODEL)
    b4_items, b4_emb, b3_items, b3_emb, b3_data_items, b3_data_id_set = (
        embed_all_items(model, item_to_factor)
    )

    # ------------------------------------------------------------------
    # Steps 4-5: Greedy pairing
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("[4-5/10] Cascade pairing and theoretical factor assignment...")
    paired_df, unpaired_b3_ids = pair_and_assign(
        b4_items, b4_emb, b3_items, b3_emb, item_to_factor, OUTDIR, model=model
    )
    print(f"  Factor distribution of paired B3 items:")
    for factor in FACTOR_ORDER:
        n = int((paired_df["b4_factor"] == factor).sum())
        in_data = int(paired_df.loc[paired_df["b4_factor"] == factor,
                                    "b3_item_id"].isin(b3_data_id_set).sum())
        print(f"    {_abbrev(factor):6s}: {n:3d} total   {in_data:3d} within 34-68")

    # ------------------------------------------------------------------
    # Step 6a: Visual UC centroid (display anchor only)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("[6a/10] Computing visual UC centroid (display anchor)...")
    visual_unnamed_centroid = compute_visual_unnamed_centroid(
        b3_items, b3_emb, unpaired_b3_ids
    )

    # ------------------------------------------------------------------
    # Step 6b: EF factor centroids (3 methods, cross-loader-aware)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("[6b/10] Computing EF factor centroids (3 methods, cross-loader-aware)...")
    all_ef_centroids = compute_centroids(
        item_to_factor, item_to_all_factors,
        b4_items, b4_emb, paired_df, b3_items, b3_emb
    )

    # ------------------------------------------------------------------
    # Step 6c: LOO assignment over ALL unpaired B3 items
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("[6c/10] LOO centroid assignment (ALL unpaired B3 items)...")

    b3_id_to_idx = {row["item_id"]: i for i, row in b3_items.iterrows()}
    paired_b3_ids = set(paired_df["b3_item_id"])

    # ALL unpaired B3 items (not just dHCP)
    unpaired_all_b3 = (
        b3_items[b3_items["item_id"].isin(unpaired_b3_ids)]
        .reset_index(drop=True)
    )
    unpaired_all_emb = np.vstack(
        [b3_emb[b3_id_to_idx[iid]] for iid in unpaired_all_b3["item_id"]]
    )
    print(f"  Total unpaired B3 items eligible for LOO: {len(unpaired_all_b3)}")

    # dHCP 34-68 items not covered by pairing (for model spec)
    unpaired_34_68 = (
        b3_data_items[~b3_data_items["item_id"].isin(paired_b3_ids)]
        .reset_index(drop=True)
    )
    print(f"  Unpaired dHCP items (34-68) eligible for LOO: {len(unpaired_34_68)}")

    all_centroid_assignments_34_68: dict[str, pd.DataFrame] = {}
    all_full_assignments: dict[str, pd.DataFrame] = {}
    all_uc_profiles: dict[str, pd.DataFrame] = {}
    all_ca_all: dict[str, pd.DataFrame] = {}   # all 91 unpaired B3 assignments

    for method in CENTROID_METHODS:
        print(f"\n  [{method}]")

        # --- Run LOO over ALL unpaired B3 ---
        ca_all = assign_via_centroids_loo(
            unpaired_all_b3,
            unpaired_all_emb,
            all_ef_centroids[method],
            b3_emb,
            b3_id_to_idx,
            unpaired_b3_ids,
        )
        ca_all.to_csv(OUTDIR / method / "all_unpaired_b3_assignments.csv", index=False)
        all_ca_all[method] = ca_all

        # --- Filter to dHCP 34-68 for model spec ---
        ca_dhcp = ca_all[ca_all["item_id"].isin(b3_data_id_set)].reset_index(drop=True)
        ca_dhcp.to_csv(OUTDIR / method / "unpaired_34_68_assignments.csv", index=False)

        n_uc = int((ca_dhcp["assigned_factor"] == UNNAMED_CLUSTER).sum())
        for factor in FACTOR_ORDER:
            n = int((ca_dhcp["assigned_factor"] == factor).sum())
            if n:
                print(f"    {_abbrev(factor):6s}: {n} item(s) assigned to EF factor")
        if n_uc:
            print(f"    {_abbrev(UNNAMED_CLUSTER):6s}: {n_uc} item(s) LOO-assigned to UC")

        all_centroid_assignments_34_68[method] = ca_dhcp
        all_full_assignments[method] = build_full_34_68_assignment(
            paired_df, ca_dhcp, b3_data_items
        )

    # ------------------------------------------------------------------
    # Step 7: Save model specs (with UC factor + sparse EF merging)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("[7/10] Saving model specifications to config/model_specs.yaml...")
    saved_names = []
    for method in CENTROID_METHODS:
        name, sparse_ef, excluded_items, redirected_items = save_model_spec(
            all_full_assignments[method],
            all_centroid_assignments_34_68[method],
            b3_data_items,
            method,
            paired_df,
            OUTDIR,
        )
        saved_names.append(name)

        # --- UC similarity profile ---
        uc_prof = compute_uc_profile(
            all_centroid_assignments_34_68[method],
            b3_data_items,
            sparse_ef,
            excluded_items,
            redirected_items,
            OUTDIR,
            method,
        )
        all_uc_profiles[method] = uc_prof
        n_redir = len(uc_prof[uc_prof["source"] == "loo_uc_redirected"])
        n_excl = len(uc_prof[uc_prof["source"] == "sparse_ef_excluded"])
        print(f"    UC profile: {len(uc_prof)} items "
              f"({n_redir} LOO-UC redirected, {n_excl} sparse-EF excluded)")

    # ------------------------------------------------------------------
    # Step 8: Visualizations
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("[8/10] Generating visualizations...")

    plot_pairing_mds(b4_items, b4_emb, b3_items, b3_emb, paired_df, OUTDIR)
    plot_pairing_three_subplots(
        b4_items, b4_emb, b3_items, b3_emb, paired_df, b3_data_id_set,
        all_ef_centroids,
        all_centroid_assignments_34_68,
        visual_unnamed_centroid,
        OUTDIR,
    )
    _mixed_ca_all = all_ca_all.get("mixed_centroids")
    plot_b3_factor_umap(b3_items, b3_emb, paired_df, b3_data_id_set, OUTDIR,
                        ca_all_df=_mixed_ca_all)
    plot_b3_factor_dr_variants(b3_items, b3_emb, paired_df, b3_data_id_set, OUTDIR,
                               ca_all_df=_mixed_ca_all)
    for method, ca in all_centroid_assignments_34_68.items():
        plot_centroid_similarity_heatmap(ca, method, OUTDIR)
    for method, ca in all_ca_all.items():
        plot_centroid_similarity_heatmap(ca, method, OUTDIR,
                                         fig_prefix="03b", item_scope="all 52 unpaired")
    plot_assignment_comparison(all_centroid_assignments_34_68, OUTDIR)
    plot_factor_sizes(paired_df, b3_data_id_set, all_full_assignments, OUTDIR)
    plot_uc_similarity_profile(all_uc_profiles, OUTDIR)

    # New figures: 01c, 01d, 07 (using mixed_centroids)
    _method = "mixed_centroids"
    if _method in all_centroid_assignments_34_68:
        _ca_dhcp = all_centroid_assignments_34_68[_method]
        _ef_cen = all_ef_centroids[_method]
        plot_b3_dhcp_uc_redirected(
            b3_items=b3_items,
            b3_emb=b3_emb,
            paired_df=paired_df,
            b3_data_id_set=b3_data_id_set,
            ef_centroids=_ef_cen,
            centroid_assignments_34_68=_ca_dhcp,
            visual_unnamed_centroid=visual_unnamed_centroid,
            outdir=OUTDIR,
            ca_all_df=all_ca_all.get(_method),
        )

    print(f"\n{'=' * 70}")
    print("[9/10] Generating UC decision logic figures (07, 07b)...")
    plot_uc_decision_logic(
        all_centroid_assignments_34_68,
        all_uc_profiles,
        OUTDIR,
    )
    plot_uc_decision_logic_all(
        all_ca_all,
        b3_data_id_set,
        OUTDIR,
    )

    # ------------------------------------------------------------------
    # Step 10: Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("[10/10] PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"Output directory : {OUTDIR}")
    print(f"model_specs.yaml : {MODEL_SPECS_FILE}")
    print("Saved model keys :")
    for n in saved_names:
        print(f"  {n}")
    print()


if __name__ == "__main__":
    main()
