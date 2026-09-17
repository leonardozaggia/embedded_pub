"""Supplementary Figures S1 & S2 -- centroid-similarity profiles of the unpaired items.

S1: the 16 unpaired Bayley-III items within the dHCP-administered range (34-68);
S2: all 52 Bayley-III items without a Bayley-4 counterpart.  Each heatmap shows
the cosine similarity of an item to the five domain centroids and to the
unclassified cluster (UC), the leave-one-out assignment, and the items flagged
for expert review (UC margin > 0.10).

The pipeline itself (`bayley_nlp/pipelines/step1_translation.py`,
`plot_centroid_similarity_heatmap`) draws the same heatmaps with a
pipeline-internal title; this standalone script reproduces the exact same
heatmap from the committed assignment CSVs
(data/outputs/step1_translation/mixed_centroids/) with a clean title and a
plain-language symbol key.  It imports nothing from the pipeline package.

Run (from the repository root):
    python figures/supplementary/figS1_S2_centroid_heatmaps.py
Outputs:
    results/figures/supplementary/figS1_centroid_heatmap_dhcp_range.png
    results/figures/supplementary/figS2_centroid_heatmap_all52.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- constants copied verbatim from step1_translation.py -------------------
UNNAMED_CLUSTER = "uncategorized_cognition"
FACTOR_ABBREV = {
    "attention": "ATT",
    "working_memory": "WM",
    "goal_directed_problem_solving": "GDPS",
    "flexibility_shift": "FS",
    "higher_order_processing": "HOP",
    UNNAMED_CLUSTER: "UC",
}
FACTOR_COLORS = {UNNAMED_CLUSTER: "#888888"}


def _abbrev(f):
    return FACTOR_ABBREV.get(f, f[:6].upper())


_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _REPO_ROOT / "data/outputs/step1_translation/mixed_centroids"
OUT_DIR = _REPO_ROOT / "results/figures/supplementary"

# Factor abbreviations spelled out, exactly as the manuscript resolves them.
GLOSSARY = ("ATT Attention · WM Working Memory · GDPS Goal-Directed Problem Solving"
            " · FS Flexibility/Shift · HOP Higher-Order Processing · UC uncategorised")


def plot_heatmap(csv_path: Path, out_path: Path, title: str) -> None:
    ca = pd.read_csv(csv_path)
    sim_cols = [c for c in ca.columns if c.startswith("sim_")]
    factor_abbrevs = [c.replace("sim_", "") for c in sim_cols]
    item_ids = ca["item_id"].tolist()
    sim_matrix = ca[sim_cols].values.astype(float)
    assigned_abbrevs = [FACTOR_ABBREV.get(f, f) for f in ca["assigned_factor"]]

    # UC-redirected items belong to best_ef in the model — ★ marks that column.
    model_abbrevs_raw = [
        FACTOR_ABBREV.get(bef, bef)
        if af == UNNAMED_CLUSTER and pd.notna(bef) and bef else FACTOR_ABBREV.get(af, af)
        for af, bef in zip(ca["assigned_factor"], ca["best_ef_factor"])
    ]

    sort_key = list(zip(model_abbrevs_raw, -ca["centroid_similarity"].values))
    order = sorted(range(len(item_ids)), key=lambda i: sort_key[i])
    sim_matrix = sim_matrix[order]
    item_ids_s = [item_ids[i] for i in order]
    assigned_s = [assigned_abbrevs[i] for i in order]
    model_s = [model_abbrevs_raw[i] for i in order]

    fig_h = max(5, len(item_ids) * 0.55)
    fig, ax = plt.subplots(figsize=(len(factor_abbrevs) * 2.0, fig_h))
    im = ax.imshow(sim_matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Cosine similarity")

    uc_abbrev = _abbrev(UNNAMED_CLUSTER)
    for i, (asgn, model_asgn) in enumerate(zip(assigned_s, model_s)):
        is_uc_redirected = model_asgn != asgn
        for j, val in enumerate(sim_matrix[i]):
            star = "★" if factor_abbrevs[j] == model_asgn else ""
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
    ax.set_yticklabels([f"{iid} → {ms}" for iid, ms in zip(item_ids_s, model_s)],
                       fontsize=10, fontfamily="monospace")

    # Clean, jargon-free title + a plain-language symbol key.
    ax.set_title(
        f"{title}\n"
        "★ = factor assigned to the item"
        "   ·   * = assigned to its best-fitting factor (nearest centroid was UC)\n"
        + GLOSSARY,
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out_path.name}  ({len(item_ids)} items)")


def main() -> None:
    plot_heatmap(
        DATA_DIR / "unpaired_34_68_assignments.csv",
        OUT_DIR / "figS1_centroid_heatmap_dhcp_range.png",
        "Centroid similarity heatmap — unpaired items in the dHCP range (items 34–68)",
    )
    plot_heatmap(
        DATA_DIR / "all_unpaired_b3_assignments.csv",
        OUT_DIR / "figS2_centroid_heatmap_all52.png",
        "Centroid similarity heatmap — all 52 unpaired items",
    )


if __name__ == "__main__":
    main()
