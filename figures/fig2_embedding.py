"""
Figure 2 — Semantic translation and embedding geometry of the Bayley-III items.

Three panels:
  A  Pairing raincloud — full-text cosine similarity of the 39 Bayley-4 → Bayley-III
     matched pairs, coloured by pairing stage (title-locked vs full-text fallback).
  B  UMAP of all 91 Bayley-III cognitive items, coloured by domain (locked
     palette). The manuscript uses the 2-D projection (default); --umap 3d
     renders the earlier 3-D "hero" variant.
  C  Within- vs between-domain mean cosine similarity per domain — its factor-coloured
     y-axis labels double as the colour key for panel B (no separate UMAP legend).

Requires the UMAP cache: run  python figures/_umap_coords.py  first.
This script only READS the cached coordinates — the panel-B cluster
separation / spread is controlled in _umap_coords.py (--target-weight,
--min-dist); change it there and re-run that script to rebuild the cache.

Usage:
    python figures/fig2_embedding.py --export-panel-c      # manuscript Figure 2 + panel-C table
    python figures/fig2_embedding.py --panels              # + results/figures/submission/figure2_{A,B,C,combined}.*
    python figures/fig2_embedding.py --umap 3d [--elev 20 --azim 250]   # 3-D variant
        [--color detected|theoretical]
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
import _embedding as emb

MAIN_DIR = ROOT / "results" / "figures" / "main"
SUBMISSION_DIR = ROOT / "results" / "figures" / "submission"


def _load_df():
    df = emb.load_umap_cache(ROOT)
    # The cached 'theoretical' column leaves UC items unresolved (Unclassified).
    # Overwrite it with the translated model (mixed_centroids, UC → nearest
    # cognitive factor) so panel B colours and panel C counts match the
    # manuscript distribution (ATT 21/WM 19/GDPS 26/FS 18/HOP 7).
    df = df.copy()
    df["theoretical"] = df["item_id"].map(emb.translated_short_map(ROOT))
    return df


def build(umap="3d", elev=55.0, azim=113.0, color_col="theoretical",
          df=None, hulls=False):
    """Assemble Figure 2. Pass a pre-computed `df` (with x/y/z/x2/y2 + theoretical)
    to use a custom layout (e.g. the supervised variant); otherwise the cached
    unsupervised coords are loaded. `hulls` toggles the panel-B cluster ellipses."""
    ls.set_lancet_rcparams(base=8.0)
    if df is None:
        df = _load_df()

    fig = plt.figure(figsize=(ls.mm(183), ls.mm(170)))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 2.3], width_ratios=[1.3, 1.0],
                          hspace=0.36, wspace=0.16,
                          left=0.055, right=0.975, top=0.945, bottom=0.075)

    wb_solution = "detected" if color_col == "detected" else "theoretical"

    axA = fig.add_subplot(gs[0, :])
    emb.draw_raincloud(axA, ROOT)

    # Panel B first, then C — so C's factor y-labels (the colour key for B) render
    # on top and are never occluded by B's axes.
    # Extended items (all centroid-added items beyond the 39 direct pairs) get a
    # soft grey halo so they can be eyeballed vs the directly-paired items.
    contour = emb.extended_item_ids(ROOT)
    if umap == "3d":
        axB = fig.add_subplot(gs[1, 0], projection="3d")
        emb.draw_umap3d(axB, df, color_col=color_col, elev=elev, azim=azim,
                        legend=True, hulls=hulls, contour_ids=contour)
    else:
        axB = fig.add_subplot(gs[1, 0])
        emb.draw_umap2d(axB, df, color_col=color_col, legend=True, hulls=hulls,
                        contour_ids=contour)

    axC = fig.add_subplot(gs[1, 1])
    emb.draw_within_between(axC, ROOT, solution=wb_solution)


    if umap == "3d":
        pB = axB.get_position()
        axB.set_position([pB.x0 - 0.050, pB.y0 - 0.020,
                          pB.width - 0.010, pB.height + 0.030])
        axB.set_box_aspect(None, zoom=1.0)
    else:
        # 2-D: pull panel B's right edge in so its scatter clears panel C's
        # factor labels (which sit just left of panel C).
        pB = axB.get_position()
        axB.set_position([pB.x0, pB.y0, pB.width - 0.075, pB.height])

    # ── Panel letters (B aligned with C on the shared row top) ────────────
    ls.panel_label(axA, "A", x=-0.01, y=1.05)
    posB, posC = axB.get_position(), axC.get_position()
    letter_y = max(posB.y1, posC.y1) + 0.012
    fig.text(posB.x0 - 0.008, letter_y, "B", fontsize=12, fontweight="bold",
             va="baseline", ha="right")
    fig.text(posC.x0 - 0.028, letter_y, "C", fontsize=12, fontweight="bold",
             va="baseline", ha="right")
    return fig


def build_panels(color_col="theoretical", df=None, hulls=False) -> dict:
    """Panels A, B and C of the 2-D figure as separate figures, drawn by the
    same functions as the combined figure."""
    ls.set_lancet_rcparams(base=8.0)
    if df is None:
        df = _load_df()
    wb_solution = "detected" if color_col == "detected" else "theoretical"
    contour = emb.extended_item_ids(ROOT)
    panels = {}

    figA = plt.figure(figsize=(ls.mm(183), ls.mm(55)))
    axA = figA.add_axes([0.055, 0.20, 0.92, 0.75])
    emb.draw_raincloud(axA, ROOT)
    panels["A"] = figA

    figB = plt.figure(figsize=(ls.mm(100), ls.mm(105)))
    axB = figB.add_axes([0.09, 0.08, 0.88, 0.89])
    emb.draw_umap2d(axB, df, color_col=color_col, legend=True, hulls=hulls,
                    contour_ids=contour)
    panels["B"] = figB

    figC = plt.figure(figsize=(ls.mm(92), ls.mm(105)))
    axC = figC.add_axes([0.26, 0.09, 0.71, 0.88])
    emb.draw_within_between(axC, ROOT, solution=wb_solution)
    panels["C"] = figC
    return panels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umap", choices=["3d", "2d"], default="2d")
    ap.add_argument("--elev", type=float, default=55.0)
    ap.add_argument("--azim", type=float, default=113.0)
    ap.add_argument("--color", choices=["detected", "theoretical"], default="theoretical")
    ap.add_argument("--export-panel-c", action="store_true",
                    help="Export panel-C numeric summary table as CSV.")
    ap.add_argument("--panel-c-out", default=None,
                    help="Output CSV path for panel-C table (default: results/tables/fig2_panelC_<solution>.csv).")
    ap.add_argument("--panels", action="store_true",
                    help="also write the individual panels and a copy of the combined "
                         "figure to results/figures/submission/ (figure2_A/B/C, "
                         "figure2_combined); 2-D layout only")
    args = ap.parse_args()
    fig = build(args.umap, args.elev, args.azim, args.color)
    stem = "fig2_embedding" + ("" if args.umap == "2d" else "_3d")
    ls.save_figure(fig, stem, MAIN_DIR)
    if args.panels:
        if args.umap != "2d":
            raise SystemExit("--panels is only defined for the manuscript's 2-D layout")
        ls.save_submission(2, stem, MAIN_DIR, build_panels(args.color), SUBMISSION_DIR)
    if args.export_panel_c:
        wb_solution = "detected" if args.color == "detected" else "theoretical"
        table = emb.panel_c_table(ROOT, solution=wb_solution)
        out_csv = Path(args.panel_c_out) if args.panel_c_out else ROOT / "results" / "tables" / f"fig2_panelC_{wb_solution}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_csv, index=False)
        print(f"Panel C table exported: {out_csv}")
    print(f"Figure 2 (embedding geometry, {args.umap}) done.")


if __name__ == "__main__":
    main()
