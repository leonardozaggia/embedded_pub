import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

"""
figS7_within_vs_between.py  (Supplementary Figure S7)
==============================
Cleveland dot plot (lollipop variant) showing, for each factor:

  • Within-cluster mean pairwise cosine similarity  (large filled circle + ±1 SD error bar)
  • Between-cluster mean pairwise cosine similarity to EACH OTHER factor
    (smaller open circles, coloured by the other factor)

A thin horizontal guide line spans from the minimum between-cluster value
to the within-cluster mean so the gap is immediately readable.

Two panels: the translated structure (Objective 1) and the bottom-up NLP solution (Objective 3).
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines  as mlines
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parents[2]
EMB_DIR = ROOT / "data/outputs/step3_b3_bottomup/all_mpnet_base_v2_kmeans_consensus_no_dr"
OUT     = ROOT / "results/figures/supplementary"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "figures"))
import lancet_style as ls   # locked palette/typography, shared with the main-text figures
import _embedding as _emb   # translated / bottom-up item->domain maps (single source of truth)
ls.set_lancet_rcparams(base=8.0)

# ── constants ─────────────────────────────────────────────────────────────────
FACTOR_SHORT = {
    "attention":                     "ATT",
    "working_memory":                "WM",
    "goal_directed_problem_solving": "GDPS",
    "flexibility_shift":             "FS",
    "higher_order_processing":       "HOP",
}
FACTOR_COLORS = {k: ls.FACTOR_COLORS[k] for k in ("ATT", "WM", "GDPS", "FS", "HOP")}
FACTOR_ORDER = ["ATT", "WM", "GDPS", "FS", "HOP"]

# ── load ──────────────────────────────────────────────────────────────────────
items_b3 = pd.read_csv(EMB_DIR / "items.csv")
emb_b3   = np.load(EMB_DIR / "embeddings_all_mpnet_base_v2.npy")
norms    = np.linalg.norm(emb_b3, axis=1, keepdims=True)
emb_n    = emb_b3 / np.maximum(norms, 1e-10)
sim_full = emb_n @ emb_n.T

# ── factor assignments ────────────────────────────────────────────────────────
# Translated structure = results/tables/bayley3_item_domain_assignments.csv
# (cascade pairing + centroid assignment + expert review); bottom-up solution =
# consensus clusters labelled post hoc (Objective 3).  Both maps are the ones
# used by Figure 2C and Figure 5B, so the panels cannot drift apart.
theo_all = items_b3[["item_id"]].copy()
theo_all["fs"] = theo_all["item_id"].map(_emb.translated_short_map(ROOT))
nlp_all = items_b3[["item_id"]].copy()
nlp_all["fs"] = nlp_all["item_id"].map(_emb.detected_short_map(ROOT))

# ── similarity helpers ────────────────────────────────────────────────────────
def within_stats(fa, fs_arr):
    idx = np.where(fs_arr == fa)[0]
    if len(idx) < 2:
        return np.nan, np.nan
    block = sim_full[np.ix_(idx, idx)]
    vals  = block[np.triu_indices(len(idx), k=1)]
    return float(vals.mean()), float(vals.std())


def between_mean(fa, fb, fs_arr):
    ia = np.where(fs_arr == fa)[0]
    ib = np.where(fs_arr == fb)[0]
    if len(ia) == 0 or len(ib) == 0:
        return np.nan
    return float(sim_full[np.ix_(ia, ib)].mean())


# ── figure ────────────────────────────────────────────────────────────────────
solutions = [
    ("Translated (Objective 1)",   theo_all["fs"].values),
    ("Bottom-up NLP (Objective 3)", nlp_all["fs"].values),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
fig.patch.set_facecolor("white")

for ax, (sol_name, fs_arr) in zip(axes, solutions):
    present = [f for f in FACTOR_ORDER if (fs_arr == f).any()]
    n_fac   = len(present)
    y_map   = {f: (n_fac - 1 - i) for i, f in enumerate(present)}

    for foc in present:
        yi  = y_map[foc]
        col = FACTOR_COLORS[foc]
        n_i = int((fs_arr == foc).sum())
        w_mean, w_sd = within_stats(foc, fs_arr)

        bvs = {o: between_mean(foc, o, fs_arr)
               for o in present if o != foc}
        valid_b = [v for v in bvs.values() if not np.isnan(v)]

        # row band
        ax.axhspan(yi - 0.45, yi + 0.45, color="0.97", lw=0, zorder=0)

        # guide line: min(between) → within mean
        if valid_b and not np.isnan(w_mean):
            ax.hlines(yi, min(valid_b), w_mean,
                      color="0.75", lw=1.2, zorder=1)

        # between-cluster dots (coloured by other factor)
        for other, bv in bvs.items():
            if np.isnan(bv):
                continue
            ax.scatter(bv, yi, s=75, color=FACTOR_COLORS[other],
                       marker="o", edgecolors="white", linewidths=0.8,
                       alpha=0.90, zorder=3)

        # within-cluster dot
        if not np.isnan(w_mean):
            ax.scatter(w_mean, yi, s=180, color=col,
                       edgecolors="black", linewidths=1.1, zorder=5)

        # factor label (coloured, bold) to the left
        ax.text(-0.015, yi, f"{foc}  (n={n_i})",
                transform=ax.get_yaxis_transform(),
                ha="right", va="center",
                fontsize=10, fontweight="bold", color=col)

    ax.set_yticks([])
    ax.set_xlabel("Mean pairwise cosine similarity", fontsize=10)
    ax.set_title(sol_name, fontsize=12, fontweight="bold", pad=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax.tick_params(axis="x", labelsize=9)
    ax.set_ylim(-0.6, n_fac - 0.4)
    ax.grid(axis="x", color="0.88", lw=0.6, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)

# ── legend ────────────────────────────────────────────────────────────────────
within_h  = mlines.Line2D([], [], marker="o", color="0.35",
                           markeredgecolor="black", markeredgewidth=1.1,
                           markersize=11, linestyle="None",
                           label="Within-cluster (mean)")
between_h = mlines.Line2D([], [], marker="o", color="0.35",
                           markeredgecolor="white", markeredgewidth=0.8,
                           markersize=8, linestyle="None", alpha=0.9,
                           label="Between-cluster\n(coloured by other factor)")
spacer    = mlines.Line2D([], [], linestyle="none", label=" ")
factor_hs = [
    mlines.Line2D([], [], marker="o", color=FACTOR_COLORS[f],
                  markersize=8, linestyle="None", label=f)
    for f in FACTOR_ORDER
]

axes[1].legend(
    handles=[within_h, between_h, spacer] + factor_hs,
    fontsize=8.5, loc="upper right",
    framealpha=0.95, edgecolor="0.80",
    handlelength=1.0, handletextpad=0.6, labelspacing=0.4,
)

fig.suptitle(
    "Within- vs between-cluster cosine similarity by factor",
    fontsize=13, fontweight="bold", y=1.01,
)

plt.tight_layout(w_pad=3.5)
ls.save_figure(fig, "figS7_within_vs_between", OUT, formats=("pdf", "png", "svg"), dpi=300)
plt.close(fig)
