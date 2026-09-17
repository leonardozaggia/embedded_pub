
# --- UTF-8 stdout/stderr, matching the rest of the figure suite ---
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")
del _stream

"""
fig_threeway_correspondence.py
================================
Manuscript in-text "Figure 6" — three-way correspondence, for the Bayley-4
item set, across:

  A  Theoretical (expert, Aylward et al. 2022) vs Detected (unsupervised
     consensus k-means)                          ARI = 0·49, NMI = 0·63
  B  Detected vs Empirical (Aylward et al. 2022 normative CFA structure)
                                                   ARI = 0·42, NMI = 0·56
  C  Theoretical vs Empirical -- the direct agreement between Aylward's own
     two reference models, cited in the Results as the "realistic upper
     bound" any automated method could be expected to reach.
                                                   ARI = 0·19, NMI = 0·40

Data source: the canonical Bayley-4 consensus-clustering run at
  data/outputs/step2_b4_validation/all_mpnet_base_v2_kmeans_consensus_k_precomputed_no_dr/
This is the run whose ARI/NMI reproduce the manuscript-reported values exactly
(confirmed by cross-checking comparison_metrics.json / empirical_vs_theoretical_metrics.json
against the Results text before writing this script) -- the sibling
"..._k_precomputed" run (with dimensionality reduction) gives close but not
identical values and is NOT the one cited in the text.

Row/column contingency logic (multi-factor item expansion, canonical display
ordering, name canonicalisation) is reused from
analysis/concordance/b4_publication_heatmaps.py so the three panels here
are guaranteed to match the counts in that script's standalone PNGs -- this
script only changes the rendering (lancet_style) and layout (one composite
three-panel figure instead of three separate files).

Usage:
    python figures/supplementary/figS6_threeway_correspondence.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "figures"))
import lancet_style as ls
import _validation as val   # panel A: identical data + drawer to Figure 5A

sys.path.insert(0, str(ROOT / "analysis" / "concordance"))
import b4_publication_heatmaps as ph  # reuse data-wrangling helpers verbatim (panels B, C)

RUN_DIR = ROOT / "data" / "outputs" / "step2_b4_validation" / "all_mpnet_base_v2_kmeans_consensus_k_precomputed_no_dr"
CLUSTER_FILE = RUN_DIR / "03_clustering" / "cluster_assignments.csv"
LABELS_FILE = RUN_DIR / "04_labeling" / "cluster_concept_labels.csv"
REF_MODELS_FILE = ROOT / "config" / "reference_models_b4.yaml"

# Manuscript-reported ARI/NMI (verified to match the .json metrics in
# RUN_DIR/05_model_comparison/{comparison_metrics,empirical_vs_theoretical_metrics}.json
# to 2 decimals -- see script docstring).
ARI_NMI = {
    "theo_det": (0.49, 0.63),
    "det_emp":  (0.42, 0.56),
    "theo_emp": (0.19, 0.40),
}


def load_empirical_data():
    """Panels B, C: anything touching the empirical (factor_1..factor_5) reference,
    which has no semantic name, so it can't reuse _validation's WM/FS/GDPS/ATT/HOP
    ORDER machinery -- mirrors publication_heatmaps.py's own data path instead."""
    detected_df = pd.read_csv(CLUSTER_FILE)
    all_items = detected_df["item_id"].tolist()
    detected_labels = detected_df["cluster"].to_numpy(dtype=int)

    detected_names, _ = ph.load_cluster_labels(LABELS_FILE)
    reference_models = ph.load_reference_models(REF_MODELS_FILE)

    theoretical_dict = {k: v for k, v in reference_models["theoretical"].items() if isinstance(v, list)}
    empirical_dict = {k: v for k, v in reference_models["empirical"].items() if isinstance(v, list)}
    theoretical_dict = ph.reorder_theoretical_dict(theoretical_dict)

    theoretical_names = ph.map_name_list(list(theoretical_dict.keys()), ph.THEORETICAL_NAME_MAP)
    # short labels ("E1".."E5") -- "Factor 1".."Factor 5" overflows the narrow
    # sub-panel width and the tick labels visually collide.
    empirical_names = [f"E{i}" for i in range(1, len(empirical_dict) + 1)]
    detected_names = ph.map_name_list(detected_names, ph.DETECTED_NAME_MAP)
    detected_names_display, det_old_to_new = ph._reorder_detected_to_canonical(detected_names)

    theoretical_map = ph.convert_factor_dict_to_item_factor_map(theoretical_dict, all_items)
    empirical_map = ph.convert_factor_dict_to_item_factor_map(empirical_dict, all_items)

    det_em, em_exp = ph.expand_labels_for_multi_factor(all_items, detected_labels, empirical_map)
    valid_em = (em_exp >= 0) & (det_em >= 0)
    mat_det_emp = ph._compute_contingency_matrix(
        ph._apply_permutation(det_em[valid_em], det_old_to_new), em_exp[valid_em],
        len(detected_names_display), len(empirical_names))

    theo_pair, emp_pair = ph.expand_pair_labels_for_multi_factor(all_items, theoretical_map, empirical_map)
    valid_pair = (theo_pair >= 0) & (emp_pair >= 0)
    mat_theo_emp = ph._compute_contingency_matrix(
        theo_pair[valid_pair], emp_pair[valid_pair],
        len(theoretical_names), len(empirical_names))

    return {
        "theoretical_names": theoretical_names,
        "detected_names": detected_names_display,
        "empirical_names": empirical_names,
        "mat_det_emp": mat_det_emp,
        "mat_theo_emp": mat_theo_emp,
    }


def draw_confusion_labelled(ax, M, row_labels, col_labels, row_colors, col_colors,
                            row_axis="", col_axis="", cmap="Blues"):
    """Same visual language as _validation.draw_confusion, generalised to
    non-square matrices and arbitrary (non-ORDER) row/col label lists --
    needed here because the empirical reference factors have no semantic name."""
    n_rows, n_cols = M.shape
    vmax = M.max() if M.max() > 0 else 1
    ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
    for i in range(n_rows):
        for j in range(n_cols):
            v = M[i, j]
            if v == 0:
                continue
            dark = v > 0.6 * vmax
            ax.text(j, i, str(int(v)), ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color="white" if dark else "#1a1a1a")
    ax.set_xticks(range(n_cols)); ax.set_yticks(range(n_rows))
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticklabels(row_labels, fontsize=7.5)
    for tick, name in zip(ax.get_xticklabels(), col_labels):
        tick.set_color(col_colors.get(name, "#333333")); tick.set_fontweight("bold")
    for tick, name in zip(ax.get_yticklabels(), row_labels):
        tick.set_color(row_colors.get(name, "#333333")); tick.set_fontweight("bold")
    ax.set_xlabel(col_axis, fontsize=8.5)
    ax.set_ylabel(row_axis, fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)


def _ari_nmi_caption(ax, ari, nmi):
    ax.text(0.5, -0.32, f"ARI = {ls.fmt_num(ari, 2)}, NMI = {ls.fmt_num(nmi, 2)}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            fontweight="bold", color="#222222")


def build():
    ls.set_lancet_rcparams(base=8.0)
    root = ROOT

    # Panel A: identical matrix + drawer to the manuscript's Figure 5A (Bayley-4,
    # theoretical vs detected) -- guarantees the two figures never disagree on
    # the confusion counts. NOTE: val.load_b4()'s own returned (ari, nmi) is a
    # *different* statistic than the manuscript's published one -- it's computed
    # on a single "primary factor per item" label, whereas the published ARI=0.49
    # /NMI=0.63 comes from the multi-factor-expanded comparison in
    # bayley_nlp/cli/05_compare_models.py (see comparison_metrics.json). This is
    # exactly why fig5_validation.py hardcodes its STATS dict instead of trusting
    # the live value -- we do the same here via ARI_NMI["theo_det"] below.
    mat_theo_det, _live_ari_a, _live_nmi_a, _n_a = val.load_b4(root)

    d = load_empirical_data()

    def cmap_for(names):
        return {n: ls.factor_color(n) for n in names}

    det_colors = cmap_for(d["detected_names"])
    theo_colors = {f: ls.FACTOR_COLORS[f] for f in val.ORDER}
    # empirical factors have no semantic name (factor_1..factor_5) -> neutral slate
    emp_colors = {n: ls.UC_COLOR for n in d["empirical_names"]}

    fig, axes = plt.subplots(1, 3, figsize=(ls.mm(183), ls.mm(72)))

    val.draw_confusion(axes[0], mat_theo_det,
                       row_axis="Theoretical factors", col_axis="Detected clusters")
    _ari_nmi_caption(axes[0], *ARI_NMI["theo_det"])

    draw_confusion_labelled(axes[1], d["mat_det_emp"], d["detected_names"], d["empirical_names"],
                            det_colors, emp_colors,
                            row_axis="Detected clusters", col_axis="Empirical factors")
    _ari_nmi_caption(axes[1], *ARI_NMI["det_emp"])

    draw_confusion_labelled(axes[2], d["mat_theo_emp"], d["theoretical_names"], d["empirical_names"],
                            theo_colors, emp_colors,
                            row_axis="Theoretical factors", col_axis="Empirical factors")
    _ari_nmi_caption(axes[2], *ARI_NMI["theo_emp"])

    for ax, letter in zip(axes, "ABC"):
        ls.panel_label(ax, letter, x=-0.30, y=1.08)

    fig.suptitle(
        "Three-way correspondence: theoretical, detected, and empirical Bayley-4 domain structures",
        fontsize=10.5, fontweight="bold", y=1.04,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.30, wspace=0.85)
    return fig


def main():
    fig = build()
    ls.save_figure(fig, "figS6_threeway_correspondence", ROOT / "results" / "figures" / "supplementary")
    print("Figure 6 (three-way theoretical/detected/empirical correspondence) done.")


if __name__ == "__main__":
    main()
