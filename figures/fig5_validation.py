"""
Figure 5 — Unsupervised validation of the domain structure.

Two parallel comparisons, each as a confusion heatmap + native (vector) Sankey:
  A  Bayley-4  (39 items) — unsupervised clusters vs the expert-derived structure.
  B  Bayley-III (91 items) — bottom-up clusters vs the translated (Step 1) structure.

The confusion matrices are computed from the committed assignment CSVs and
reproduce the project's confusion figures exactly. ARI/NMI annotations are read
from results/metrics/manuscript_metrics.json (written by the
analysis/concordance scripts), so they are the same numbers as in the text.

Usage:
    python figures/fig5_validation.py            # results/figures/main/fig5_validation.*
    python figures/fig5_validation.py --panels   # + results/figures/submission/figure5_{A,B,combined}.*
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent

import lancet_style as ls
import _validation as val

METRICS_JSON = ROOT / "results" / "metrics" / "manuscript_metrics.json"
MAIN_DIR = ROOT / "results" / "figures" / "main"
SUBMISSION_DIR = ROOT / "results" / "figures" / "submission"
STEM = "fig5_validation"
COMBINED_H_MM = 166.0
HEADER_MM = 0.045 * COMBINED_H_MM      # title baseline above the heatmap axes top
STATS_MM = 0.022 * COMBINED_H_MM       # ARI/NMI line below the title baseline

ROWS = {
    "A": dict(loader=val.load_b4, row_axis="Expert-derived factor", col_axis="Unsupervised cluster",
              title="Bayley-4 (39 items): expert-derived vs unsupervised",
              caption="Expert-derived → Unsupervised", key="b4"),
    "B": dict(loader=val.load_b3, row_axis="Translated factor", col_axis="Bottom-up cluster",
              title="Bayley-III (91 items): translated vs bottom-up",
              caption="Translated → Bottom-up", key="b3"),
}


def _fmt(x: float, nd: int = 2) -> str:
    """0.4936 -> '0·49' (Lancet mid-line decimal)."""
    return f"{x:.{nd}f}".replace(".", "·")


def load_stats() -> dict:
    """Concordance statistics from results/metrics/manuscript_metrics.json, written
    by analysis/concordance/{b4,b3}_sankey.py and *_item_level_analysis.py.

    Bayley-4: ARI/NMI from the multi-factor-expanded labelling of
    bayley_nlp/cli/05_compare_models.py (the values reported in the paper:
    0·49 / 0·63; 24/39 items, 61·5%).  Bayley-III: bottom-up clusters vs the
    translated structure (0·39 / 0·48; 31/91 items, 34·1%).
    """
    import json
    if not METRICS_JSON.exists():
        raise SystemExit(f"Missing {METRICS_JSON.relative_to(ROOT)} -- run the "
                         "analysis/concordance scripts first (see docs/REPRODUCING.md)")
    m = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    out = {}
    for key, sankey, item in (("b4", "Bayley4_Sankey_Theo_vs_Det", "Bayley4_ItemLevel_Theo_vs_Det"),
                              ("b3", "Bayley3_Sankey_Theo_vs_Det", "Bayley3_ItemLevel_Theo_vs_Det")):
        out[key] = dict(
            ari=_fmt(m[sankey]["ARI"]), nmi=_fmt(m[sankey]["NMI"]),
            agree=f"{m[item]['overall_agreement_fraction']} ({_fmt(m[item]['overall_agreement_rate_pct'], 1)}%)",
        )
    return out


def _load():
    return {"stats": load_stats(),
            "M": {p: ROWS[p]["loader"](ROOT)[0] for p in ROWS}}


def draw_row(fig, spec, panel: str, data):
    """Confusion heatmap + Sankey of one comparison into the gridspec slot `spec`;
    returns (heatmap axes, sankey axes)."""
    cfg, M = ROWS[panel], data["M"][panel]
    cols = [ls.FACTOR_COLORS[f] for f in val.ORDER]
    sub = spec.subgridspec(1, 2, width_ratios=[0.82, 1.18], wspace=0.10)
    ax0 = fig.add_subplot(sub[0, 0])
    ax1 = fig.add_subplot(sub[0, 1])
    val.draw_confusion(ax0, M, row_axis=cfg["row_axis"], col_axis=cfg["col_axis"])
    ls.sankey(ax1, M, val.ORDER, val.ORDER, cols, cols, ribbon_alpha=0.5)
    ax1.set_title(cfg["caption"], fontsize=8.5, fontweight="bold", color="#333",
                  pad=2, loc="center")
    return ax0, ax1


def draw_header(fig, ax0, panel: str, data, letter: bool = True):
    """Section header (letter, title, ARI/NMI line) above the heatmap axes."""
    s = data["stats"][ROWS[panel]["key"]]
    h_mm = fig.get_figheight() * 25.4
    y = ax0.get_position().y1 + HEADER_MM / h_mm
    if letter:
        fig.text(0.012, y, panel, fontsize=13, fontweight="bold", va="baseline")
    fig.text(0.05, y, ROWS[panel]["title"], fontsize=10.5, fontweight="bold", va="baseline")
    fig.text(0.05, y - STATS_MM / h_mm, f"ARI = {s['ari']}   NMI = {s['nmi']}",
             fontsize=8, color="#555", va="baseline")


def build(data=None):
    ls.set_lancet_rcparams(base=8.0)
    data = data or _load()
    fig = plt.figure(figsize=(ls.mm(183), ls.mm(COMBINED_H_MM)))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.42,
                          left=0.09, right=0.985, top=0.90, bottom=0.065)
    axA0, _ = draw_row(fig, gs[0], "A", data)
    axB0, _ = draw_row(fig, gs[1], "B", data)
    draw_header(fig, axA0, "A", data)
    draw_header(fig, axB0, "B", data)
    return fig


def build_panels(data=None) -> dict:
    """Each comparison (heatmap + Sankey) as its own figure."""
    ls.set_lancet_rcparams(base=8.0)
    data = data or _load()
    panels = {}
    for panel in ROWS:
        fig = plt.figure(figsize=(ls.mm(183), ls.mm(90)))
        gs = fig.add_gridspec(1, 1, left=0.09, right=0.985, top=0.78, bottom=0.12)
        ax0, _ = draw_row(fig, gs[0], panel, data)
        draw_header(fig, ax0, panel, data, letter=False)
        panels[panel] = fig
    return panels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", action="store_true",
                    help="also write the individual panels and a copy of the combined "
                         "figure to results/figures/submission/ (figure5_A, figure5_B, "
                         "figure5_combined)")
    args = ap.parse_args()
    data = _load()
    fig = build(data)
    ls.save_figure(fig, STEM, MAIN_DIR)
    if args.panels:
        ls.save_submission(5, STEM, MAIN_DIR, build_panels(data), SUBMISSION_DIR)
    print("Figure 5 (validation) done.")


if __name__ == "__main__":
    main()
