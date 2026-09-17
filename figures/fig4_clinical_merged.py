"""
Figure 4: Clinical figure — MERGED variant.

Combines the covariate (MIMIC) and outcome associations into a single two-panel
figure (A = covariates, B = outcomes) sharing the four factor columns. This is
the slot-saving layout: using it as one figure frees a slot for the standalone
3D UMAP hero (fig2_umap_hero.py).

Reuses _clinical.draw_block, so it is guaranteed identical in grammar to the
separate Fig 3 / Fig 4.

Usage:
    python figures/fig4_clinical_merged.py            # results/figures/main/fig4_clinical_merged.*
    python figures/fig4_clinical_merged.py --panels   # + results/figures/submission/figure4_{A,B,combined}.*
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
import _clinical as cl

MAIN_DIR = ROOT / "results" / "figures" / "main"
SUBMISSION_DIR = ROOT / "results" / "figures" / "submission"
STEM = "fig4_clinical_merged"

# Vertical offsets of the furniture around each block, in mm (they were tuned
# as fractions of the 268 mm combined figure; expressing them in mm keeps the
# single-panel figures identical in spacing).
COMBINED_H_MM = 268.0
TITLE_MM = {"A": 0.050 * COMBINED_H_MM, "B": 0.048 * COMBINED_H_MM}
SUBTITLE_MM = {"A": 0.032 * COMBINED_H_MM, "B": 0.030 * COMBINED_H_MM}
XLABEL_MM = {"A": 0.032 * COMBINED_H_MM, "B": 0.030 * COMBINED_H_MM}
TITLES = {"A": "Perinatal, neonatal and sociodemographic covariates",
          "B": "Developmental and behavioural outcomes"}
SUBTITLES = {"A": "Filled = p < ·05; open = not significant.  Bold label = significant "
                  "factor contrast (DIFFTEST p < ·05).  Continuous covariates fully "
                  "standardised; binary covariates semistandardised.",
             "B": "Latent correlations from the full measurement model."}
XLABELS = {"A": "Standardised association (β)", "B": "Correlations with latent factors (r)"}
HEADER_H = {"A": 0.072, "B": 0.11}     # factor-header height, fraction of the axes height (draw_block)


def _load():
    mimic_df, mk, mmeta, sig_codes = cl.load_mimic(ROOT)
    out_df, ok, ometa, _ = cl.load_outcomes(ROOT)
    return {"A": (mimic_df, mk, mmeta, sig_codes), "B": (out_df, ok, ometa, set())}


def _fy(fig, mm_: float) -> float:
    """mm -> figure-fraction (vertical)."""
    return mm_ / (fig.get_figheight() * 25.4)


def draw_block(fig, spec, panel: str, data):
    """One four-column association block (A = covariates, B = outcomes) into
    the gridspec slot `spec`; returns its four axes."""
    df, key, meta, sig_codes = data[panel]
    sub = spec.subgridspec(1, 4, wspace=0.10)
    ax0 = fig.add_subplot(sub[0, 0])
    axs = [ax0] + [fig.add_subplot(sub[0, j], sharey=ax0) for j in (1, 2, 3)]
    if panel == "A":
        cl.draw_block(axs, df, key, meta, ls.PREDICTOR_DOMAIN_COLORS,
                      metric="estimate", sig_codes=sig_codes, ylabel_fs=7.4,
                      header_h=HEADER_H["A"], header_fs=8.0)
    else:
        cl.draw_block(axs, df, key, meta, ls.OUTCOME_DOMAIN_COLORS,
                      metric="estimate", ylabel_fs=7.6, header_h=HEADER_H["B"], header_fs=8.0,
                      plain_label_color="#111111")
    for ax in axs:
        ax.set_xlabel("")
    return axs


def draw_furniture(fig, axs, panel: str, letter: bool = True,
                   title_mm: float | None = None, subtitle_mm: float | None = None):
    """Title, subtitle and centred x-axis label of one block.  `title_mm` /
    `subtitle_mm` = distance above the axes top (default: the combined-figure
    values)."""
    pos = axs[1].get_position()
    title_mm = TITLE_MM[panel] if title_mm is None else title_mm
    subtitle_mm = SUBTITLE_MM[panel] if subtitle_mm is None else subtitle_mm
    fig.text(0.61, pos.y0 - _fy(fig, XLABEL_MM[panel]), XLABELS[panel], ha="center", fontsize=9)
    if letter:
        fig.text(0.015, pos.y1 + _fy(fig, title_mm), panel, fontsize=13,
                 fontweight="bold", va="bottom")
    fig.text(0.050, pos.y1 + _fy(fig, title_mm), TITLES[panel],
             fontsize=10.5, fontweight="bold", va="bottom")
    fig.text(0.050, pos.y1 + _fy(fig, subtitle_mm), SUBTITLES[panel],
             fontsize=7.2, color="#666", va="bottom")


def draw_legend(fig, panel: str, y: float, with_significance: bool):
    if panel == "A":
        handles, title, ncol = cl.domain_legend_handles(ls.PREDICTOR_DOMAIN_COLORS), "Covariate domain (A)", 3
    else:
        handles, title, ncol = cl.domain_legend_handles(ls.OUTCOME_DOMAIN_COLORS), "Outcome domain (B)", 4
    if with_significance:
        handles = handles + cl.significance_legend_handles()
        title += "  /  significance"
        ncol += 2
    # fig.legend() registers the legend itself; adding it as an artist as well
    # would draw it twice (slightly heavier anti-aliasing).
    return fig.legend(handles=handles, title=title, loc="center", ncol=ncol,
                      frameon=False, fontsize=7.8, title_fontsize=8,
                      bbox_to_anchor=(0.55, y), handletextpad=0.4,
                      columnspacing=1.6 if panel == "A" else 1.4)


def build(data=None):
    ls.set_lancet_rcparams(base=8.0)
    data = data or _load()
    fig = plt.figure(figsize=(ls.mm(183), ls.mm(COMBINED_H_MM)))
    gs = fig.add_gridspec(2, 1, height_ratios=[len(data["A"][2]), len(data["B"][2])],
                          hspace=0.34, left=0.235, right=0.985, top=0.915, bottom=0.135)
    axesA = draw_block(fig, gs[0], "A", data)
    axesB = draw_block(fig, gs[1], "B", data)
    draw_furniture(fig, axesA, "A")
    draw_furniture(fig, axesB, "B")
    # Legends: two centred rows at the bottom (no horizontal overlap)
    draw_legend(fig, "A", 0.072, with_significance=False)
    draw_legend(fig, "B", 0.024, with_significance=True)
    return fig


def build_panels(data=None) -> dict:
    """Each block as its own figure (same drawing code, own title and legend)."""
    ls.set_lancet_rcparams(base=8.0)
    data = data or _load()
    panels = {}
    for panel, h_mm, top, bottom, leg_y in (("A", 180.0, 0.855, 0.135, 0.035),
                                            ("B", 130.0, 0.805, 0.165, 0.040)):
        fig = plt.figure(figsize=(ls.mm(183), ls.mm(h_mm)))
        gs = fig.add_gridspec(1, 1, left=0.235, right=0.985, top=top, bottom=bottom)
        axs = draw_block(fig, gs[0], panel, data)
        # the header boxes rise header_h * (axes height) above the axes top
        header_mm = HEADER_H[panel] * (top - bottom) * h_mm
        draw_furniture(fig, axs, panel, letter=False,
                       subtitle_mm=header_mm + 2.0, title_mm=header_mm + 7.0)
        draw_legend(fig, panel, leg_y, with_significance=True)
        panels[panel] = fig
    return panels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", action="store_true",
                    help="also write the individual panels and a copy of the combined "
                         "figure to results/figures/submission/ (figure4_A, figure4_B, "
                         "figure4_combined)")
    args = ap.parse_args()
    data = _load()
    fig = build(data)
    ls.save_figure(fig, STEM, MAIN_DIR)
    if args.panels:
        ls.save_submission(4, STEM, MAIN_DIR, build_panels(data), SUBMISSION_DIR)
    print("Merged clinical figure done.")


if __name__ == "__main__":
    main()
