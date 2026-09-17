import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

"""
figS5_cosine_vs_tetrachoric.py  (Supplementary Figure S5)
============================
Side-by-side heatmaps for the 23 CFA items (WM=4, GDPS=8, FS=11),
ordered by factor:

  Panel 1 — Cosine similarity (embedding space)
  Panel 2 — Tetrachoric correlation (empirical, dHCP N=739)
  Panel 3 — Difference: cosine − tetrachoric
             (positive = embedding overestimates; negative = underestimates)

Factor blocks are outlined and labelled on each panel.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = ROOT / "data/outputs/step3_b3_bottomup/all_mpnet_base_v2_kmeans_consensus_no_dr"
TET_DIR = ROOT / "data/outputs/tetrachoric"          # written by analysis/tetrachoric.R
OUT_DIR = ROOT / "results/figures/supplementary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "figures"))
import lancet_style as ls   # locked palette/typography, shared with the main-text figures
ls.set_lancet_rcparams(base=8.0)

FACTOR_SHORT = {
    "working_memory":               "WM",
    "goal_directed_problem_solving":"GDPS",
    "flexibility_shift":            "FS",
}
FACTOR_COLORS = {k: ls.FACTOR_COLORS[k] for k in ("WM", "GDPS", "FS")}

# ── load tetrachoric matrix and item order ────────────────────────────────────
tet_raw = pd.read_csv(TET_DIR / "tetrachoric_matrix.csv", index_col=0)
factor_map = pd.read_csv(TET_DIR / "cfa_item_factor_map.csv")

# ordered items as output by R (WM → GDPS → FS)
ordered_items = factor_map["item_col"].tolist()
factor_labels = factor_map["assigned_factor"].tolist()

tet_mat = tet_raw.loc[ordered_items, ordered_items].values.astype(float)
np.fill_diagonal(tet_mat, 1.0)

# ── build cosine similarity matrix for same 23 items ─────────────────────────
items_all = pd.read_csv(EMB_DIR / "items.csv")
emb_all   = np.load(EMB_DIR / "embeddings_all_mpnet_base_v2.npy")

# item_id in items_all uses underscore: COG_040; ordered_items uses COG040
id_to_idx = {row.item_id.replace("_", ""): i
             for i, row in items_all.iterrows()}
idx23 = [id_to_idx[item] for item in ordered_items]
emb23 = emb_all[idx23]

norms  = np.linalg.norm(emb23, axis=1, keepdims=True)
emb23n = emb23 / np.maximum(norms, 1e-10)
cos_mat = emb23n @ emb23n.T

diff_mat = cos_mat - tet_mat   # positive = overestimate

# ── factor block boundaries ───────────────────────────────────────────────────
factor_order = ["working_memory", "goal_directed_problem_solving", "flexibility_shift"]
factor_sizes = [factor_labels.count(f) for f in factor_order]
boundaries   = np.cumsum([0] + factor_sizes)   # [0, 4, 12, 23]

n = len(ordered_items)

# ── tick labels: short item number within factor ──────────────────────────────
tick_labels = [f"C{item[3:]}" for item in ordered_items]


def draw_factor_blocks(ax, boundaries, factor_order, factor_colors, lw=2.0):
    """Draw coloured rectangles around each factor block."""
    for i, fac in enumerate(factor_order):
        lo, hi = boundaries[i], boundaries[i + 1]
        size = hi - lo
        color = factor_colors[FACTOR_SHORT[fac]]
        rect = patches.Rectangle(
            (lo - 0.5, lo - 0.5), size, size,
            linewidth=lw, edgecolor=color, facecolor="none", zorder=5,
        )
        ax.add_patch(rect)
        ax.text(
            (lo + hi) / 2 - 0.5, lo - 1.0,
            FACTOR_SHORT[fac],
            ha="center", va="bottom", fontsize=12, fontweight="bold",
            color="black", zorder=6,
        )


def add_gridlines(ax, boundaries):
    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color="white", lw=1.0, zorder=4)
        ax.axvline(b - 0.5, color="white", lw=1.0, zorder=4)


# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 6.5))
fig.patch.set_facecolor("white")

panels = [
    (cos_mat,  "RdBu_r", -1, 1,  "Cosine similarity\n(embedding space)"),
    (tet_mat,  "RdBu_r", -1, 1,  "Tetrachoric correlation\n(empirical, dHCP)"),
    (diff_mat, "PuOr_r", -0.8, 0.8,
     "Difference\n(cosine − tetrachoric)"),
]

for ax, (mat, cmap, vmin, vmax, title) in zip(axes, panels):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="equal", interpolation="nearest")

    add_gridlines(ax, boundaries)
    draw_factor_blocks(ax, boundaries, factor_order, FACTOR_COLORS)

    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                        orientation="vertical")
    cbar.ax.tick_params(labelsize=10)

# diagonal note on difference panel
axes[2].text(
    0.5, -0.22,
    "Diagonal excluded (trivially 0). Off-diagonal positive values mean the "
    "embedding predicts\nhigher relatedness than respondents actually show.",
    transform=axes[2].transAxes, ha="center", va="top",
    fontsize=6.5, color="0.4", style="italic",
)

fig.suptitle(
    "Item similarity matrices for the 23-item CFA model (WM · GDPS · FS)\n"
    "ordered by factor — embedding cosine similarity vs empirical tetrachoric correlation",
    fontsize=14, fontweight="bold", y=1.03,
)

plt.tight_layout()
ls.save_figure(fig, "figS5_cosine_vs_tetrachoric", OUT_DIR, formats=("pdf", "png", "svg"), dpi=300)
plt.close(fig)

# ── quick summary stats ───────────────────────────────────────────────────────
triu = np.triu_indices(n, k=1)
cos_off  = cos_mat[triu]
tet_off  = tet_mat[triu]
diff_off = diff_mat[triu]

print(f"\nOff-diagonal summary (N={len(cos_off)} pairs):")
print(f"  Cosine sim    mean={cos_off.mean():.3f}  sd={cos_off.std():.3f}  range=[{cos_off.min():.3f}, {cos_off.max():.3f}]")
print(f"  Tetrachoric   mean={tet_off.mean():.3f}  sd={tet_off.std():.3f}  range=[{tet_off.min():.3f}, {tet_off.max():.3f}]")
print(f"  Difference    mean={diff_off.mean():.3f}  sd={diff_off.std():.3f}  range=[{diff_off.min():.3f}, {diff_off.max():.3f}]")
r = np.corrcoef(cos_off, tet_off)[0, 1]
print(f"  Pearson r(cosine, tetrachoric) = {r:.3f}")

# within-factor vs between-factor breakdown
within_pairs = [(i, j) for i in range(n) for j in range(i+1, n)
                if factor_labels[i] == factor_labels[j]]
between_pairs = [(i, j) for i in range(n) for j in range(i+1, n)
                 if factor_labels[i] != factor_labels[j]]

def pair_stats(pairs, mat, label):
    vals = [mat[i, j] for i, j in pairs]
    print(f"  {label}: mean={np.mean(vals):.3f}  sd={np.std(vals):.3f}  n={len(vals)}")

print("\nWithin-factor pairs:")
pair_stats(within_pairs, cos_mat,  "  Cosine    ")
pair_stats(within_pairs, tet_mat,  "  Tetrachoric")
pair_stats(within_pairs, diff_mat, "  Difference ")

print("\nBetween-factor pairs:")
pair_stats(between_pairs, cos_mat,  "  Cosine    ")
pair_stats(between_pairs, tet_mat,  "  Tetrachoric")
pair_stats(between_pairs, diff_mat, "  Difference ")
